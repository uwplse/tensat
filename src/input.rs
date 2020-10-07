use crate::model::*;
use egg::*;
use itertools::Itertools;
use std::collections::HashMap;

const MAX_DIM: usize = 8;

/// Struct for converting a model specified using our Rust interface to RecExpr
///
/// The RecExpr is growed on the fly when member functions are called. Uses a
/// Hashmap to store the map of scalar nodes to their indices into the RexExpr to
/// avoid replication.
#[derive(Default)]
pub struct GraphConverter {
    rec_expr: RecExpr<Mdl>,
    scalar_map: HashMap<i32, Id>,
    name_gen: NameGen,
}

/// Struct for storing information of a tensor. This is passed between functions
/// during graph creation.
#[derive(Copy, Clone, Default)]
pub struct TensorInfo {
    /// Id into the RecExpr constructed
    pub id: Id,
    /// Shape of the tensor. We deal with tensor up to MAX_DIM dimensions
    pub shape: [i32; MAX_DIM],
    /// Number of dimensions of this tensor
    pub n_dim: usize,
}

/// The APIs of GraphConverter are (intended to) match TASO's so that we can easily
/// construct TASO graphs using this class
impl GraphConverter {
    /// Gets the RexExpr after graph is constructed
    pub fn rec_expr(self) -> RecExpr<Mdl> {
        self.rec_expr
    }

    /// Takes in the parameters for the new input, construct the node in RexExpr,
    /// return the Id (index) of this input node in the RecExpr. This is the
    /// pattern for all these op functions.
    pub fn new_input(&mut self, dims: &[i32]) -> TensorInfo {
        let name = self.name_gen.new_input_name() + "@" + &dims.iter().join("_");
        let node = Mdl::Var(Symbol::from(name));
        let name_id = self.rec_expr.add(node);

        let new_node = Mdl::Input([name_id]);
        let (shape, n_dim) = self.shape_from_dim(dims);
        TensorInfo {
            id: self.rec_expr.add(new_node),
            shape,
            n_dim,
        }
    }

    pub fn new_weight(&mut self, dims: &[i32]) -> TensorInfo {
        let name = self.name_gen.new_weight_name() + "@" + &dims.iter().join("_");
        let node = Mdl::Var(Symbol::from(name));
        let name_id = self.rec_expr.add(node);

        let new_node = Mdl::Weight([name_id]);
        let (shape, n_dim) = self.shape_from_dim(dims);
        TensorInfo {
            id: self.rec_expr.add(new_node),
            shape,
            n_dim,
        }
    }

    pub fn conv2d(
        &mut self,
        inpt: TensorInfo,
        wght: TensorInfo,
        stride_h: i32,
        stride_w: i32,
        padding: i32,
        activation: i32,
    ) -> TensorInfo {
        let stride_h_id = self.add_or_get_val(stride_h);
        let stride_w_id = self.add_or_get_val(stride_w);
        let padding_id = self.add_or_get_val(padding);
        let activation_id = self.add_or_get_val(activation);

        let new_node = Mdl::Conv2d([
            stride_h_id,
            stride_w_id,
            padding_id,
            activation_id,
            inpt.id,
            wght.id,
        ]);

        // Get shape
        let mut shape = [0; MAX_DIM];
        let input_h = inpt.shape[2];
        let input_w = inpt.shape[3];
        let kernel_h = wght.shape[2];
        let kernel_w = wght.shape[3];

        let (output_h, output_w) = self.get_conv_shape(
            input_h, input_w, stride_h, stride_w, kernel_h, kernel_w, padding,
        );
        shape[0] = inpt.shape[0];
        shape[1] = wght.shape[0];
        shape[2] = output_h;
        shape[3] = output_w;

        TensorInfo {
            id: self.rec_expr.add(new_node),
            shape: shape,
            n_dim: 4,
        }
    }

    pub fn dropout(&mut self, inpt: TensorInfo) -> TensorInfo {
        let new_node = Mdl::Dropout(inpt.id);

        TensorInfo {
            id: self.rec_expr.add(new_node),
            ..inpt
        }
    }

    pub fn relu(&mut self, inpt: TensorInfo) -> TensorInfo {
        let new_node = Mdl::Relu(inpt.id);

        TensorInfo {
            id: self.rec_expr.add(new_node),
            ..inpt
        }
    }

    pub fn tanh(&mut self, inpt: TensorInfo) -> TensorInfo {
        let new_node = Mdl::Tanh(inpt.id);

        TensorInfo {
            id: self.rec_expr.add(new_node),
            ..inpt
        }
    }

    pub fn sigmoid(&mut self, inpt: TensorInfo) -> TensorInfo {
        let new_node = Mdl::Sigmoid(inpt.id);

        TensorInfo {
            id: self.rec_expr.add(new_node),
            ..inpt
        }
    }

    pub fn batchnorm(&mut self, inpt: TensorInfo, scale: TensorInfo, bias: TensorInfo, mean: TensorInfo, var: TensorInfo) -> TensorInfo {
        let new_node = Mdl::BatchNorm([inpt.id, scale.id, bias.id, mean.id, var.id]);
        TensorInfo {
            id: self.rec_expr.add(new_node),
            ..inpt
        }
    }

    pub fn add(&mut self, inpt_1: TensorInfo, inpt_2: TensorInfo) -> TensorInfo {
        let new_node = Mdl::Ewadd([inpt_1.id, inpt_2.id]);

        TensorInfo {
            id: self.rec_expr.add(new_node),
            ..inpt_1
        }
    }

    pub fn matmul(&mut self, inpt_1: TensorInfo, inpt_2: TensorInfo) -> TensorInfo {
        let activation = ACTNONE;
        let act_id = self.add_or_get_val(activation);

        let new_node = Mdl::Matmul([act_id, inpt_1.id, inpt_2.id]);

        let mut shape = inpt_1.shape;
        let n_dim = inpt_1.n_dim;
        shape[n_dim - 1] = inpt_2.shape[n_dim - 1];

        TensorInfo {
            id: self.rec_expr.add(new_node),
            shape,
            n_dim,
        }
    }

    pub fn mul(&mut self, inpt_1: TensorInfo, inpt_2: TensorInfo) -> TensorInfo {
        let new_node = Mdl::Ewmul([inpt_1.id, inpt_2.id]);

        TensorInfo {
            id: self.rec_expr.add(new_node),
            ..inpt_1
        }
    }

    pub fn concat(
        &mut self,
        axis: i32,
        ndim: i32,
        inpt_1: TensorInfo,
        inpt_2: TensorInfo,
    ) -> TensorInfo {
        // Only support concat of 2 inputs for now
        // To support more, pass in a slice and create more concat nodes here
        let axis_id = self.add_or_get_val(axis);
        let ndim_id = self.add_or_get_val(ndim);

        let new_node = Mdl::Concat([axis_id, ndim_id, inpt_1.id, inpt_2.id]);

        let mut shape = inpt_1.shape;
        let n_dim = inpt_1.n_dim;
        shape[axis as usize] += inpt_2.shape[axis as usize];

        TensorInfo {
            id: self.rec_expr.add(new_node),
            shape,
            n_dim,
        }
    }

    pub fn concat_multi(&mut self, axis: i32, inputs: &[TensorInfo]) -> TensorInfo {
        let n_inputs = inputs.len();
        // We can add supports for other number of inputs later when needed.
        // We need to add a new Concat op for each number of inputs
        assert!(n_inputs <= 5);

        let n_dim = inputs[0].n_dim;
        let axis_id = self.add_or_get_val(axis);
        let ndim_id = self.add_or_get_val(n_dim as i32);

        let new_node = match n_inputs {
            2 => {
                Mdl::Concat([
                   axis_id,
                   ndim_id,
                   inputs[0].id,
                   inputs[1].id,
                ])
            }
            3 => {
                Mdl::Concat3([
                   axis_id,
                   ndim_id,
                   inputs[0].id,
                   inputs[1].id,
                   inputs[2].id,
                ])
            }
            4 => {
                Mdl::Concat4([
                   axis_id,
                   ndim_id,
                   inputs[0].id,
                   inputs[1].id,
                   inputs[2].id,
                   inputs[3].id,
                ])
            }
            5 => {
                Mdl::Concat5([
                   axis_id,
                   ndim_id,
                   inputs[0].id,
                   inputs[1].id,
                   inputs[2].id,
                   inputs[3].id,
                   inputs[4].id,
                ])
            }
            _ => panic!("Number of input for concat not supported"),
        };

        let mut shape = inputs[0].shape;
        shape[axis as usize] += (1..n_inputs)
            .map(|i| inputs[i].shape[axis as usize])
            .sum::<i32>();

        TensorInfo {
            id: self.rec_expr.add(new_node),
            shape,
            n_dim,
        }
    }

    pub fn maxpool2d(
        &mut self,
        inpt: TensorInfo,
        kernel_h: i32,
        kernel_w: i32,
        stride_h: i32,
        stride_w: i32,
        padding: i32,
    ) -> TensorInfo {
        let kernel_h_id = self.add_or_get_val(kernel_h);
        let kernel_w_id = self.add_or_get_val(kernel_w);
        let stride_h_id = self.add_or_get_val(stride_h);
        let stride_w_id = self.add_or_get_val(stride_w);
        let padding_id = self.add_or_get_val(padding);
        let activation = ACTNONE;
        let act_id = self.add_or_get_val(activation);

        let new_node = Mdl::Poolmax([
            inpt.id,
            kernel_h_id,
            kernel_w_id,
            stride_h_id,
            stride_w_id,
            padding_id,
            act_id,
        ]);

        // Get shape
        let mut shape = [0; MAX_DIM];
        let input_h = inpt.shape[2];
        let input_w = inpt.shape[3];

        let (output_h, output_w) = self.get_conv_shape(
            input_h, input_w, stride_h, stride_w, kernel_h, kernel_w, padding,
        );
        shape[0] = inpt.shape[0];
        shape[1] = inpt.shape[1];
        shape[2] = output_h;
        shape[3] = output_w;

        TensorInfo {
            id: self.rec_expr.add(new_node),
            shape: shape,
            n_dim: 4,
        }
    }

    pub fn avgpool2d(
        &mut self,
        inpt: TensorInfo,
        kernel_h: i32,
        kernel_w: i32,
        stride_h: i32,
        stride_w: i32,
        padding: i32,
    ) -> TensorInfo {
        let kernel_h_id = self.add_or_get_val(kernel_h);
        let kernel_w_id = self.add_or_get_val(kernel_w);
        let stride_h_id = self.add_or_get_val(stride_h);
        let stride_w_id = self.add_or_get_val(stride_w);
        let padding_id = self.add_or_get_val(padding);
        let activation = ACTNONE;
        let act_id = self.add_or_get_val(activation);

        let new_node = Mdl::Poolavg([
            inpt.id,
            kernel_h_id,
            kernel_w_id,
            stride_h_id,
            stride_w_id,
            padding_id,
            act_id,
        ]);

        // Get shape
        let mut shape = [0; MAX_DIM];
        let input_h = inpt.shape[2];
        let input_w = inpt.shape[3];

        let (output_h, output_w) = self.get_conv_shape(
            input_h, input_w, stride_h, stride_w, kernel_h, kernel_w, padding,
        );
        shape[0] = inpt.shape[0];
        shape[1] = inpt.shape[1];
        shape[2] = output_h;
        shape[3] = output_w;

        TensorInfo {
            id: self.rec_expr.add(new_node),
            shape: shape,
            n_dim: 4,
        }
    }

    pub fn enlarge(&mut self, inpt_1: TensorInfo, inpt_2: TensorInfo) -> TensorInfo {
        let mut shape = inpt_1.shape;
        shape[2] = inpt_2.shape[2];
        shape[3] = inpt_2.shape[3];

        let new_node = Mdl::Enlarge([inpt_1.id, inpt_2.id]);

        TensorInfo {
            id: self.rec_expr.add(new_node),
            shape: shape,
            n_dim: 4,
        }
    }

    pub fn split(&mut self, axis: i32, inpt: TensorInfo) -> (TensorInfo, TensorInfo) {
        let axis_id = self.add_or_get_val(axis);

        let split_node = Mdl::Split([axis_id, inpt.id]);
        let split_id = self.rec_expr.add(split_node);
        let split_0_node = Mdl::Split0(split_id);
        let split_0_id = self.rec_expr.add(split_0_node);
        let split_1_node = Mdl::Split1(split_id);
        let split_1_id = self.rec_expr.add(split_1_node);

        assert!(false, "Shape inference not implemented for split");

        let out_0 = TensorInfo {
            id: split_0_id,
            shape: [0; MAX_DIM],
            n_dim: inpt.n_dim,
        };
        let out_1 = TensorInfo {
            id: split_1_id,
            shape: [0; MAX_DIM],
            n_dim: inpt.n_dim,
        };
        (out_0, out_1)
    }

    pub fn reshape(&mut self, inpt: TensorInfo, shape: &[i32]) -> TensorInfo {
        let shape_name = &shape.iter().join("_");
        let node = Mdl::Var(Symbol::from(shape_name));
        let shape_name_id = self.rec_expr.add(node);

        let new_node = Mdl::Reshape([inpt.id, shape_name_id]);
        let (shape_new, n_dim) = self.shape_from_dim(shape);
        TensorInfo {
            id: self.rec_expr.add(new_node),
            shape: shape_new,
            n_dim: n_dim,
        }
    }

    pub fn transpose(&mut self, inpt: TensorInfo, perm: &[i32], shuffle: bool) -> TensorInfo {
        let perm_name = &perm.iter().join("_");
        let node = Mdl::Var(Symbol::from(perm_name));
        let perm_name_id = self.rec_expr.add(node);

        let shuffle_val = if shuffle { SHUFFLE } else { NOSHUFFLE };
        let shuffle_id = self.add_or_get_val(shuffle_val);

        let new_node = Mdl::Transpose([inpt.id, perm_name_id, shuffle_id]);

        let mut shape = [0; MAX_DIM];
        let n_dim = inpt.n_dim;
        for (i, perm_i) in perm.iter().enumerate() {
            shape[i] = inpt.shape[*perm_i as usize];
        }
        TensorInfo {
            id: self.rec_expr.add(new_node),
            shape,
            n_dim,
        }
    }

    pub fn noop(&mut self, inpt_1: TensorInfo, inpt_2: TensorInfo) -> TensorInfo {
        let new_node = Mdl::Noop([inpt_1.id, inpt_2.id]);
        TensorInfo {
            id: self.rec_expr.add(new_node),
            shape: [0; MAX_DIM],
            n_dim: inpt_1.n_dim,
        }
    }

    /// If a scalar value is in the RecExpr, gets the Id. Otherwise creates one.
    fn add_or_get_val(&mut self, val: i32) -> Id {
        match self.scalar_map.get(&val) {
            Some(id) => *id,
            None => {
                let node = Mdl::Num(val);
                let id = self.rec_expr.add(node);
                self.scalar_map.insert(val, id);
                id
            }
        }
    }

    fn shape_from_dim(&self, dims: &[i32]) -> ([i32; MAX_DIM], usize) {
        let mut shape = [0; MAX_DIM];
        for (i, dim) in dims.iter().enumerate() {
            shape[i] = *dim;
        }
        (shape, dims.len())
    }

    fn get_conv_shape(
        &self,
        input_h: i32,
        input_w: i32,
        stride_h: i32,
        stride_w: i32,
        kernel_h: i32,
        kernel_w: i32,
        padding: i32,
    ) -> (i32, i32) {
        if padding == PSAME {
            let output_h = (input_h + stride_h - 1) / stride_h;
            let output_w = (input_w + stride_w - 1) / stride_w;
            (output_h, output_w)
        } else {
            let output_h = (input_h - kernel_h) / stride_h + 1;
            let output_w = (input_w - kernel_w) / stride_w + 1;
            (output_h, output_w)
        }
    }
}

/// Struct for generating new names for weight tensors in the model
///
/// Generates names like w1, w2...
#[derive(Default)]
pub struct NameGen {
    count_input: i32,
    count_weight: i32,
}

impl NameGen {
    pub fn new_weight_name(&mut self) -> String {
        let name = format!("w_{}", self.count_weight);
        self.count_weight += 1;
        name
    }

    pub fn new_input_name(&mut self) -> String {
        let name = format!("input_{}", self.count_input);
        self.count_input += 1;
        name
    }
}
