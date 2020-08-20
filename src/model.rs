#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

//use rand::prelude::*;
use rand;
use root::taso::*;
use std::collections::HashSet;
use std::convert::TryInto;
use std::time::{Duration, Instant};

use egg::*;

// Operator parameters, value matches the TASO side
pub const PSAME: i32 = 0;
pub const PVALID: i32 = 1;

pub const ACTNONE: i32 = 0;
pub const ACTSIGMOID: i32 = 1;
pub const ACTRELU: i32 = 2;
pub const ACTTANH: i32 = 3;

pub const NOSHUFFLE: i32 = 0;
pub const SHUFFLE: i32 = 1;

define_language! {
    pub enum Mdl {
        "input"     = Input([Id; 1]), // takes a Var, format: name@dim1_dim2...
        "weight"    = Weight([Id; 1]), // takes a Var, format : name@dim1_dim2...
        "ewadd"     = Ewadd([Id; 2]),
        "ewmul"     = Ewmul([Id; 2]),
        "smul"      = Smul([Id; 2]),
        "transpose" = Transpose([Id; 3]), // input, perm_name (format: dim1_dim2...), shuffle
        "matmul"    = Matmul([Id; 3]), // activation, input1, input2
        "conv2d"    = Conv2d([Id; 6]), // conv2d's weight tensor kernel size can not be even, it seems that TASO's output shape computation is incorrect for even kernal size (like 4x4)
        "enlarge"   = Enlarge([Id; 2]), // input_to_enlarge, ref_input
        "relu"      = Relu(Id),
        "tanh"      = Tanh(Id),
        "sigmoid"   = Sigmoid(Id),
        "poolavg"   = Poolavg([Id; 7]), // input, kernel_h, kernel_w, stride_h, stride_w, padding, activation
        "poolmax"   = Poolmax([Id; 7]), // input, kernel_h, kernel_w, stride_h, stride_w, padding, activation
        "concat"    = Concat([Id; 4]), // axis, ndim, input1, input2. ndim is for using in CheckApply only
        "split_0"   = Split0(Id), // must take a split node as input
        "split_1"   = Split1(Id), // must take a split node as input
        "split"     = Split([Id; 2]), // axis, input
        "Cpool"     = Cpool([Id; 2]),
        "Iconv"     = Iconv([Id; 2]),
        "Imatmul"   = Imatmul,
        "Iewmul"    = Iewmul,
        "merge"     = Merge([Id; 2]), // merge_gconv, takes [weight, count]
        "reshape"   = Reshape([Id; 2]), // input, shape_name (format: dim1_dim2...)
        "noop"      = Noop([Id; 2]), // No op, use to combine the outputs of a graph in case there are multiple, since egg works with single root graph
        Num(i32),
        Var(Symbol),
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum DataKind {
    Name,
    Scalar,
    Tnsr,
    TnsrTuple,
}

impl Default for DataKind {
    fn default() -> Self {
        DataKind::Name
    }
}

/// Metadata struct for TensorAnalysis
#[derive(Debug, Clone)]
pub struct ValTnsr {
    /// The data type of this eclass, can be a name/scalar/tensor
    pub dtype: DataKind,
    /// The value of this eclass if it is a Scalar type
    pub val: i32,
    /// The name string of this eclass if it is a Name type
    pub name: String,
    /// The pointer to the tensor if it is a Tensor type
    pub meta: TensorHandle,
    /// The pointer to the second tensor if it is a TnsrTuple type (for split node)
    pub meta_2: TensorHandle,
    /// If the tensor results from all weights computations
    pub all_weights: bool,
}

impl Default for ValTnsr {
    fn default() -> Self {
        ValTnsr {
            meta: std::ptr::null_mut(),
            meta_2: std::ptr::null_mut(),
            ..Default::default()
        }
    }
}

/// Struct for metadata analysis
///
/// In this analysis, it calls functions on the TASO side (e.g. graph.matmul())
/// to create (or get) new ops/nodes and stores pointers to the output tensors.
/// TASO will measure and store the runtime cost when creating a new op/node.
pub struct TensorAnalysis {
    /// Points to the graph object on the TASO side
    pub graph: std::cell::RefCell<Box<Graph>>,
    /// Record blacklisted nodes for filtering cycles
    pub blacklist_nodes: HashSet<Mdl>,
    /// Newly added nodes by order during single output rule application
    pub newly_added: Vec<Mdl>,
}

impl Default for TensorAnalysis {
    fn default() -> Self {
        unsafe {
            // NOTE Box heap-allocates, otherwise any pointer from
            // C++ may be dangling
            let mut graph = Box::new(Graph::new());
            Graph_Graph(&mut *graph);
            TensorAnalysis {
                graph: std::cell::RefCell::new(graph),
                blacklist_nodes: HashSet::<Mdl>::new(),
                newly_added: Vec::<Mdl>::new(),
            }
        }
    }
}

impl Analysis<Mdl> for TensorAnalysis {
    type Data = ValTnsr;

    /// Merges two metadata when two eclasses are merged.
    fn merge(&self, to: &mut Self::Data, from: Self::Data) -> bool {
        if from.all_weights && (!to.all_weights) {
            to.all_weights = from.all_weights;
            true
        } else {
            false
        }
    }

    // Constructs metadata for a new enode, using TASO side functions for tensors.
    fn make(egraph: &EGraph<Mdl, Self>, enode: &Mdl) -> Self::Data {
        let x = |i: &Id| &egraph[*i].data;
        let dim_from_name = |name: &Id| {
            let name_vec: Vec<&str> = x(name).name.split("@").collect();
            assert!(name_vec.len() == 2);
            let dims: Vec<i32> = name_vec[1]
                .split("_")
                .map(|x| x.parse::<i32>().unwrap())
                .collect();
            dims
        };

        let mut g = egraph.analysis.graph.borrow_mut();
        match enode {
            Mdl::Matmul([act, a, b]) => {
                // Check types
                assert!(x(act).dtype == DataKind::Scalar);
                assert!(x(a).dtype == DataKind::Tnsr);
                assert!(x(b).dtype == DataKind::Tnsr);

                // Get arguments
                let t_a = x(a).meta;
                let t_b = x(b).meta;
                let activation: ActiMode = x(act).val.try_into().unwrap();
                let all_weights = x(a).all_weights && x(b).all_weights;

                // Create tensorhandle and get metadata
                let res = unsafe { g.matmul(t_a, t_b, activation) };
                Self::Data {
                    dtype: DataKind::Tnsr,
                    val: 0,
                    name: String::new(),
                    meta: res,
                    meta_2: std::ptr::null_mut(),
                    all_weights: all_weights,
                }
            }

            Mdl::Conv2d([stride_h, stride_w, pad, act, inpt, wght]) => {
                // Check types
                assert!(x(stride_h).dtype == DataKind::Scalar);
                assert!(x(stride_w).dtype == DataKind::Scalar);
                assert!(x(pad).dtype == DataKind::Scalar);
                assert!(x(act).dtype == DataKind::Scalar);
                assert!(x(inpt).dtype == DataKind::Tnsr);
                assert!(x(wght).dtype == DataKind::Tnsr);

                // Get arguments
                let t_inpt = x(inpt).meta;
                let t_wght = x(wght).meta;
                let strideH = x(stride_h).val;
                let strideW = x(stride_w).val;
                let padding: PaddingMode = x(pad).val.try_into().unwrap();
                let activation: ActiMode = x(act).val.try_into().unwrap();
                let all_weights = x(inpt).all_weights && x(wght).all_weights;

                // Create tensorhandle and get metadata
                let res =
                    unsafe { g.conv2d1(t_inpt, t_wght, strideH, strideW, padding, activation) };
                Self::Data {
                    dtype: DataKind::Tnsr,
                    val: 0,
                    name: String::new(),
                    meta: res,
                    meta_2: std::ptr::null_mut(),
                    all_weights: all_weights,
                }
            }

            Mdl::Ewadd([a, b]) => {
                // Check types
                assert!(x(a).dtype == DataKind::Tnsr);
                assert!(x(b).dtype == DataKind::Tnsr);

                // Get arguments
                let t_a = x(a).meta;
                let t_b = x(b).meta;
                let all_weights = x(a).all_weights && x(b).all_weights;

                // Create tensorhandle and get metadata
                let res = unsafe { g.element(OpType_OP_EW_ADD, t_a, t_b) };
                Self::Data {
                    dtype: DataKind::Tnsr,
                    val: 0,
                    name: String::new(),
                    meta: res,
                    meta_2: std::ptr::null_mut(),
                    all_weights: all_weights,
                }
            }

            Mdl::Ewmul([a, b]) => {
                // Check types
                assert!(x(a).dtype == DataKind::Tnsr);
                assert!(x(b).dtype == DataKind::Tnsr);

                // Get arguments
                let t_a = x(a).meta;
                let t_b = x(b).meta;
                let all_weights = x(a).all_weights && x(b).all_weights;

                // Create tensorhandle and get metadata
                let res = unsafe { g.element(OpType_OP_EW_MUL, t_a, t_b) };
                Self::Data {
                    dtype: DataKind::Tnsr,
                    val: 0,
                    name: String::new(),
                    meta: res,
                    meta_2: std::ptr::null_mut(),
                    all_weights: all_weights,
                }
            }

            Mdl::Relu(a) => {
                assert!(x(a).dtype == DataKind::Tnsr);
                let t_a = x(a).meta;
                let all_weights = x(a).all_weights;

                let res = unsafe { g.relu(t_a, true) };
                Self::Data {
                    dtype: DataKind::Tnsr,
                    val: 0,
                    name: String::new(),
                    meta: res,
                    meta_2: std::ptr::null_mut(),
                    all_weights: all_weights,
                }
            }

            Mdl::Tanh(a) => {
                assert!(x(a).dtype == DataKind::Tnsr);
                let t_a = x(a).meta;
                let all_weights = x(a).all_weights;

                let res = unsafe { g.tanh(t_a, true) };
                Self::Data {
                    dtype: DataKind::Tnsr,
                    val: 0,
                    name: String::new(),
                    meta: res,
                    meta_2: std::ptr::null_mut(),
                    all_weights: all_weights,
                }
            }

            Mdl::Sigmoid(a) => {
                assert!(x(a).dtype == DataKind::Tnsr);
                let t_a = x(a).meta;
                let all_weights = x(a).all_weights;

                let res = unsafe { g.sigmoid(t_a, true) };
                Self::Data {
                    dtype: DataKind::Tnsr,
                    val: 0,
                    name: String::new(),
                    meta: res,
                    meta_2: std::ptr::null_mut(),
                    all_weights: all_weights,
                }
            }

            Mdl::Input([name]) => {
                // Check types
                assert!(x(name).dtype == DataKind::Name);

                // Get arguments
                let mut dims = dim_from_name(name);
                let ndim = dims.len();
                dims.shrink_to_fit();
                assert!(dims.len() == dims.capacity());
                let ptr = dims.as_mut_ptr();
                std::mem::forget(dims);

                // Create tensorhandle and get metadata
                let res = unsafe { g.new_input(ndim.try_into().unwrap(), ptr) };
                Self::Data {
                    dtype: DataKind::Tnsr,
                    val: 0,
                    name: String::new(),
                    meta: res,
                    meta_2: std::ptr::null_mut(),
                    all_weights: false,
                }
            }

            Mdl::Weight([name]) => {
                // Check types
                assert!(x(name).dtype == DataKind::Name);

                // Get arguments
                let mut dims = dim_from_name(name);
                let ndim = dims.len();
                dims.shrink_to_fit();
                assert!(dims.len() == dims.capacity());

                let num_entries = dims.iter().product();
                let mut weight_data: Vec<f32> = (0..num_entries).map(|_| rand::random()).collect();
                weight_data.shrink_to_fit();
                assert!(weight_data.len() == weight_data.capacity());

                let ptr = dims.as_mut_ptr();
                std::mem::forget(dims);
                let data_ptr = weight_data.as_mut_ptr();
                std::mem::forget(weight_data);

                // Create tensorhandle and get metadata
                let res = unsafe { g.new_weight(ndim.try_into().unwrap(), ptr, data_ptr) };
                Self::Data {
                    dtype: DataKind::Tnsr,
                    val: 0,
                    name: String::new(),
                    meta: res,
                    meta_2: std::ptr::null_mut(),
                    all_weights: true,
                }
            }

            Mdl::Concat([axis, ndim, a, b]) => {
                // Check types
                assert!(x(axis).dtype == DataKind::Scalar);
                assert!(x(ndim).dtype == DataKind::Scalar);
                assert!(x(a).dtype == DataKind::Tnsr);
                assert!(x(b).dtype == DataKind::Tnsr);

                // Get arguments
                let t_a = x(a).meta;
                let t_b = x(b).meta;
                let axis_val = x(axis).val;
                let all_weights = x(a).all_weights && x(b).all_weights;

                // Create tensorhandle and get metadata
                let t = [t_a, t_b];
                let res = unsafe { g.concat(axis_val, 2, t.as_ptr()) };
                Self::Data {
                    dtype: DataKind::Tnsr,
                    val: 0,
                    name: String::new(),
                    meta: res,
                    meta_2: std::ptr::null_mut(),
                    all_weights: all_weights,
                }
            }

            Mdl::Merge([weight, count]) => {
                // Check types
                assert!(x(count).dtype == DataKind::Scalar);
                assert!(x(weight).dtype == DataKind::Tnsr);

                // Get arguments
                let t_weight = x(weight).meta;
                let count_val = x(count).val;
                let all_weights = x(weight).all_weights;

                // Create tensorhandle and get metadata
                let res = unsafe { g.merge_gconv(t_weight, count_val) };
                Self::Data {
                    dtype: DataKind::Tnsr,
                    val: 0,
                    name: String::new(),
                    meta: res,
                    meta_2: std::ptr::null_mut(),
                    all_weights: all_weights,
                }
            }

            Mdl::Poolmax([inpt, kernel_h, kernel_w, stride_h, stride_w, pad, act]) => {
                // Check types
                assert!(x(kernel_h).dtype == DataKind::Scalar);
                assert!(x(kernel_w).dtype == DataKind::Scalar);
                assert!(x(stride_h).dtype == DataKind::Scalar);
                assert!(x(stride_w).dtype == DataKind::Scalar);
                assert!(x(pad).dtype == DataKind::Scalar);
                assert!(x(act).dtype == DataKind::Scalar);
                assert!(x(inpt).dtype == DataKind::Tnsr);

                // Get arguments
                let t_inpt = x(inpt).meta;
                let kernelH = x(kernel_h).val;
                let kernelW = x(kernel_w).val;
                let strideH = x(stride_h).val;
                let strideW = x(stride_w).val;
                let padding: PaddingMode = x(pad).val.try_into().unwrap();
                let activation: ActiMode = x(act).val.try_into().unwrap();
                let all_weights = x(inpt).all_weights;

                // Create tensorhandle and get metadata
                let res = unsafe {
                    g.pool2d_max(
                        t_inpt, kernelH, kernelW, strideH, strideW, padding, activation,
                    )
                };
                Self::Data {
                    dtype: DataKind::Tnsr,
                    val: 0,
                    name: String::new(),
                    meta: res,
                    meta_2: std::ptr::null_mut(),
                    all_weights: all_weights,
                }
            }

            Mdl::Split([axis, inpt]) => {
                // Check types
                assert!(x(axis).dtype == DataKind::Scalar);
                assert!(x(inpt).dtype == DataKind::Tnsr);

                // Get arguments
                let t_inpt = x(inpt).meta;
                let axis_val = x(axis).val;
                let all_weights = x(inpt).all_weights;

                // Create tensorhandle and get metadata
                unsafe {
                    // Has to do it this way since TASO side does not provide a
                    // Graph.split() function that infers split position from input
                    let op = (*g.model).get_or_create_split1(t_inpt, axis_val, 2);
                    assert!(op != Op_INVALID_OP);
                    g.add_edge((*t_inpt).op, op, (*t_inpt).idx, 0);
                    let x1 = Box::new((*op.ptr).outputs[0].clone());
                    let res_1 = Box::into_raw(x1);
                    (*res_1).op = op;
                    let x2 = Box::new((*op.ptr).outputs[1].clone());
                    let res_2 = Box::into_raw(x2);
                    (*res_2).op = op;
                    Self::Data {
                        dtype: DataKind::TnsrTuple,
                        val: 0,
                        name: String::new(),
                        meta: res_1,
                        meta_2: res_2,
                        all_weights: all_weights,
                    }
                }
            }

            Mdl::Split0(inpt) => {
                // Check types
                assert!(x(inpt).dtype == DataKind::TnsrTuple);
                let all_weights = x(inpt).all_weights;

                let res = x(inpt).meta;
                Self::Data {
                    dtype: DataKind::Tnsr,
                    val: 0,
                    name: String::new(),
                    meta: res,
                    meta_2: std::ptr::null_mut(),
                    all_weights: all_weights,
                }
            }

            Mdl::Split1(inpt) => {
                // Check types
                assert!(x(inpt).dtype == DataKind::TnsrTuple);
                let all_weights = x(inpt).all_weights;

                let res = x(inpt).meta_2;
                Self::Data {
                    dtype: DataKind::Tnsr,
                    val: 0,
                    name: String::new(),
                    meta: res,
                    meta_2: std::ptr::null_mut(),
                    all_weights: all_weights,
                }
            }

            Mdl::Enlarge([a, b]) => {
                // Check types
                assert!(x(a).dtype == DataKind::Tnsr);
                assert!(x(b).dtype == DataKind::Tnsr);

                // Get arguments
                let t_a = x(a).meta;
                let t_b = x(b).meta;
                let all_weights = x(a).all_weights && x(b).all_weights;

                // Create tensorhandle and get metadata
                let res = unsafe { g.enlarge(t_a, t_b) };
                Self::Data {
                    dtype: DataKind::Tnsr,
                    val: 0,
                    name: String::new(),
                    meta: res,
                    meta_2: std::ptr::null_mut(),
                    all_weights: all_weights,
                }
            }

            Mdl::Reshape([inpt, shape_name]) => {
                // Check types
                assert!(x(shape_name).dtype == DataKind::Name);
                assert!(x(inpt).dtype == DataKind::Tnsr);

                // Get arguments
                let dims: Vec<i32> = x(shape_name)
                    .name
                    .split("_")
                    .map(|x| x.parse::<i32>().unwrap())
                    .collect();
                let t_inpt = x(inpt).meta;
                let all_weights = x(inpt).all_weights;

                // Create tensorhandle and get metadata
                let res = unsafe {
                    let cpp_dims = convert_to_cpp_vec(&dims);
                    let ptr = cpp_dims.as_ptr() as *const [u64; 3];
                    g.reshape(t_inpt, ptr)
                };
                Self::Data {
                    dtype: DataKind::Tnsr,
                    val: 0,
                    name: String::new(),
                    meta: res,
                    meta_2: std::ptr::null_mut(),
                    all_weights: all_weights,
                }
            }

            Mdl::Transpose([inpt, perm_name, shuffle]) => {
                // Check types
                assert!(x(perm_name).dtype == DataKind::Name);
                assert!(x(inpt).dtype == DataKind::Tnsr);
                assert!(x(shuffle).dtype == DataKind::Scalar);

                // Get arguments
                let perms: Vec<i32> = x(perm_name)
                    .name
                    .split("_")
                    .map(|x| x.parse::<i32>().unwrap())
                    .collect();
                let t_inpt = x(inpt).meta;
                let shuffle_val = x(shuffle).val;
                let shuffle_bool = (shuffle_val == SHUFFLE);
                let all_weights = x(inpt).all_weights;

                // Create tensorhandle and get metadata
                let res = unsafe {
                    let cpp_perms = convert_to_cpp_vec(&perms);
                    let ptr = cpp_perms.as_ptr() as *const [u64; 3];
                    g.transpose(t_inpt, ptr, shuffle_bool)
                };
                Self::Data {
                    dtype: DataKind::Tnsr,
                    val: 0,
                    name: String::new(),
                    meta: res,
                    meta_2: std::ptr::null_mut(),
                    all_weights: all_weights,
                }
            }

            Mdl::Noop([a, b]) => {
                // Check types
                assert!(x(a).dtype == DataKind::Tnsr);
                assert!(x(b).dtype == DataKind::Tnsr);
                let all_weights = x(a).all_weights && x(b).all_weights;

                Self::Data {
                    dtype: DataKind::Tnsr,
                    val: 0,
                    name: String::new(),
                    meta: std::ptr::null_mut(),
                    meta_2: std::ptr::null_mut(),
                    all_weights: all_weights,
                }
            }

            Mdl::Num(_n) => Self::Data {
                dtype: DataKind::Scalar,
                val: *_n,
                name: String::new(),
                meta: std::ptr::null_mut(),
                meta_2: std::ptr::null_mut(),
                all_weights: false,
            },

            Mdl::Var(_s) => Self::Data {
                dtype: DataKind::Name,
                val: 0,
                name: _s.as_str().to_string(),
                meta: std::ptr::null_mut(),
                meta_2: std::ptr::null_mut(),
                all_weights: false,
            },

            other => {
                println!("{:?}", other);
                todo!()
            }
        }
    }

    // Not needed to modify anything
    fn modify(egraph: &mut EGraph<Mdl, Self>, id: Id) {}
}

/// Convert rust vector to C++ vector, for ffi
///
/// The returned C++ format for vector is:
/// [pointer_to_first_element, pointer_to_last_element, pointer_to_the_end_of_vector_capacity]
unsafe fn convert_to_cpp_vec(v: &Vec<i32>) -> [*const i32; 3] {
    [
        v.as_ptr(),
        v.as_ptr().offset(v.len().try_into().unwrap()),
        v.as_ptr().offset(v.capacity().try_into().unwrap()),
    ]
}
