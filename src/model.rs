#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

//use rand::prelude::*;
use root::taso::*;
use std::convert::TryInto;
use std::time::{Duration, Instant};
use rand;

use egg::*;

// Operator parameters, value matches the TASO side
pub const PSAME: i32 = 0;
pub const PVALID: i32 = 1;

pub const ACTNONE: i32 = 0;
pub const ACTSIGMOID: i32 = 1;
pub const ACTRELU: i32 = 2;
pub const ACTTANH: i32 = 3;

define_language! {
    pub enum Mdl {
        "input"     = Input([Id; 1]), // takes a Var, format: name@dim1_dim2...
        "weight"    = Weight([Id; 1]), // takes a Var, format : name@dim1_dim2...
        "ewadd"     = Ewadd([Id; 2]),
        "ewmul"     = Ewmul([Id; 2]),
        "smul"      = Smul([Id; 2]),
        "transpose" = Transpose(Id),
        "matmul"    = Matmul([Id; 3]), // activation, input1, input2
        "conv2d"    = Conv2d([Id; 6]), // conv2d's weight tensor kernel size can not be even, it seems that TASO's output shape computation is incorrect for even kernal size (like 4x4)
        "enlarge"   = Enlarge([Id; 3]),
        "relu"      = Relu(Id),
        "tanh"      = Tanh(Id),
        "sigmoid"   = Sigmoid(Id),
        "poolavg"   = Poolavg([Id; 6]),
        "poolmax"   = Poolmax([Id; 6]),
        "concat"    = Concat([Id; 4]), // axis, ndim, input1, input2. ndim is for using in CheckApply only
        "split_0"   = Split0([Id; 2]),
        "split_1"   = Split1([Id; 2]),
        "Cpool"     = Cpool([Id; 2]),
        "Iconv"     = Iconv([Id; 2]),
        "Imatmul"   = Imatmul,
        "Iewmul"    = Iewmul,
        "merge"     = Merge([Id; 2]), // merge_gconv, takes [weight, count]
        Num(i32),
        Var(Symbol),
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum DataKind {
    Name,
    Scalar,
    Tnsr,
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
}

impl Default for ValTnsr {
    fn default() -> Self {
        ValTnsr {
            meta: std::ptr::null_mut(),
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
            }
        }
    }
}

impl Analysis<Mdl> for TensorAnalysis {
    type Data = ValTnsr;

    /// Merges two metadata when two eclasses are merged. Because the useful
    /// parts of the metadata of two equivalent eclasses are always the same,
    /// we don't need to change
    fn merge(&self, to: &mut Self::Data, from: Self::Data) -> bool {
        false
    }

    // Constructs metadata for a new enode, using TASO side functions for tensors.
    fn make(egraph: &EGraph<Mdl, Self>, enode: &Mdl) -> Self::Data {
        let x = |i: &Id| &egraph[*i].data;
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

                // Create tensorhandle and get metadata
                unsafe {
                    let mm = g.matmul(t_a, t_b, activation);
                    Self::Data {
                        dtype: DataKind::Tnsr,
                        val: 0,
                        name: String::new(),
                        meta: mm,
                    }
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

                // Create tensorhandle and get metadata
                unsafe {
                    let res = g.conv2d1(t_inpt, t_wght, strideH, strideW, padding, activation);
                    Self::Data {
                        dtype: DataKind::Tnsr,
                        val: 0,
                        name: String::new(),
                        meta: res,
                    }
                }
            }

            Mdl::Ewadd([a, b]) => {
                // Check types
                assert!(x(a).dtype == DataKind::Tnsr);
                assert!(x(b).dtype == DataKind::Tnsr);

                // Get arguments
                let t_a = x(a).meta;
                let t_b = x(b).meta;

                // Create tensorhandle and get metadata
                unsafe {
                    let res = g.element(OpType_OP_EW_ADD, t_a, t_b);
                    Self::Data {
                        dtype: DataKind::Tnsr,
                        val: 0,
                        name: String::new(),
                        meta: res,
                    }
                }
            }

            Mdl::Ewmul([a, b]) => {
                // Check types
                assert!(x(a).dtype == DataKind::Tnsr);
                assert!(x(b).dtype == DataKind::Tnsr);

                // Get arguments
                let t_a = x(a).meta;
                let t_b = x(b).meta;

                // Create tensorhandle and get metadata
                unsafe {
                    let res = g.element(OpType_OP_EW_MUL, t_a, t_b);
                    Self::Data {
                        dtype: DataKind::Tnsr,
                        val: 0,
                        name: String::new(),
                        meta: res,
                    }
                }
            }

            Mdl::Relu(a) => {
                assert!(x(a).dtype == DataKind::Tnsr);
                let t_a = x(a).meta;

                unsafe {
                    let res = g.relu(t_a, true);
                    Self::Data {
                        dtype: DataKind::Tnsr,
                        val: 0,
                        name: String::new(),
                        meta: res,
                    }
                }
            }

            Mdl::Tanh(a) => {
                assert!(x(a).dtype == DataKind::Tnsr);
                let t_a = x(a).meta;

                unsafe {
                    let res = g.tanh(t_a, true);
                    Self::Data {
                        dtype: DataKind::Tnsr,
                        val: 0,
                        name: String::new(),
                        meta: res,
                    }
                }
            }

            Mdl::Sigmoid(a) => {
                assert!(x(a).dtype == DataKind::Tnsr);
                let t_a = x(a).meta;

                unsafe {
                    let res = g.sigmoid(t_a, true);
                    Self::Data {
                        dtype: DataKind::Tnsr,
                        val: 0,
                        name: String::new(),
                        meta: res,
                    }
                }
            }

            Mdl::Input([name]) => {
                // Check types
                assert!(x(name).dtype == DataKind::Name);

                // Get shape
                let mut split = x(name).name.split("@");
                let name_vec: Vec<&str> = split.collect();
                assert!(name_vec.len() == 2);
                let mut split_dims = name_vec[1].split("_");
                let dim_str_vec: Vec<&str> = split_dims.collect();

                // Create tensorhandle and get metadata
                unsafe {
                    let mut dims: Vec<i32> = dim_str_vec.iter().map(|x| x.parse::<i32>().unwrap()).collect();
                    let ndim = dims.len();
                    assert!(ndim <= 4);
                    dims.shrink_to_fit();
                    assert!(dims.len() == dims.capacity());
                    let ptr = dims.as_mut_ptr();
                    std::mem::forget(dims);

                    let res = g.new_input(ndim.try_into().unwrap(), ptr);
                    Self::Data {
                        dtype: DataKind::Tnsr,
                        val: 0,
                        name: String::new(),
                        meta: res,
                    }
                }
            }

            Mdl::Weight([name]) => {
                // Check types
                assert!(x(name).dtype == DataKind::Name);

                // Get shape
                let mut split = x(name).name.split("@");
                let name_vec: Vec<&str> = split.collect();
                assert!(name_vec.len() == 2);
                let mut split_dims = name_vec[1].split("_");
                let dim_str_vec: Vec<&str> = split_dims.collect();

                // Create tensorhandle and get metadata
                unsafe {
                    let mut dims: Vec<i32> = dim_str_vec.iter().map(|x| x.parse::<i32>().unwrap()).collect();
                    let ndim = dims.len();
                    assert!(ndim <= 4);
                    dims.shrink_to_fit();
                    assert!(dims.len() == dims.capacity());

                    // Create data for weight
                    let mut num_entries = 1;
                    for d in &dims {
                        num_entries = num_entries * d;
                    }
                    let mut weight_data: Vec<f32> = Vec::with_capacity(num_entries.try_into().unwrap());
                    for _ in 0..weight_data.capacity() {
                        weight_data.push(rand::random());
                    }
                    assert!(weight_data.len() == weight_data.capacity());

                    let ptr = dims.as_mut_ptr();
                    std::mem::forget(dims);
                    let data_ptr = weight_data.as_mut_ptr();
                    std::mem::forget(weight_data);

                    let res = g.new_weight(ndim.try_into().unwrap(), ptr, data_ptr);
                    Self::Data {
                        dtype: DataKind::Tnsr,
                        val: 0,
                        name: String::new(),
                        meta: res,
                    }
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

                // Create tensorhandle and get metadata
                unsafe {
                    let res = g.concat(axis_val, 2, vec![t_a, t_b].as_ptr());
                    Self::Data {
                        dtype: DataKind::Tnsr,
                        val: 0,
                        name: String::new(),
                        meta: res,
                    }
                }
            }

            Mdl::Merge([weight, count]) => {
                // Check types
                assert!(x(count).dtype == DataKind::Scalar);
                assert!(x(weight).dtype == DataKind::Tnsr);

                // Get arguments
                let t_weight = x(weight).meta;
                let count_val = x(count).val;

                // Create tensorhandle and get metadata
                unsafe {
                    let res = g.merge_gconv(t_weight, count_val);
                    Self::Data {
                        dtype: DataKind::Tnsr,
                        val: 0,
                        name: String::new(),
                        meta: res,
                    }
                }
            }

            Mdl::Num(_n) => Self::Data {
                dtype: DataKind::Scalar,
                val: *_n,
                name: String::new(),
                meta: std::ptr::null_mut(),
            },

            Mdl::Var(_s) => Self::Data {
                dtype: DataKind::Name,
                val: 0,
                name: _s.as_str().to_string(),
                meta: std::ptr::null_mut(),
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
