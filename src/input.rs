use crate::model::*;
use egg::*;
use std::collections::HashMap;

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

/// The APIs of GraphConverter are (intended to) match TASO's so that we can easily
/// constructing TASO graphs using this class
impl GraphConverter {
    /// Gets the RexExpr after graph is constructed
    pub fn rec_expr(self) -> RecExpr<Mdl> {
        self.rec_expr
    }

    /// Takes in the parameters for the new input, construct the node in RexExpr,
    /// return the Id (index) of this input node in the RecExpr. This is the
    /// pattern for all these op functions.
    pub fn new_input(&mut self, name: &str, dim1: i32, dim2: i32, dim3: i32, dim4: i32) -> Id {
        let node = Mdl::Var(Symbol::from(name));
        let name_id = self.rec_expr.add(node);
        let dim1_id = self.add_or_get_val(dim1);
        let dim2_id = self.add_or_get_val(dim2);
        let dim3_id = self.add_or_get_val(dim3);
        let dim4_id = self.add_or_get_val(dim4);

        let new_node = Mdl::Input([name_id, dim1_id, dim2_id, dim3_id, dim4_id]);
        self.rec_expr.add(new_node)
    }

    pub fn new_weight(&mut self, dim1: i32, dim2: i32, dim3: i32, dim4: i32) -> Id {
        let name = &self.name_gen.new_name();
        let node = Mdl::Var(Symbol::from(name));
        let name_id = self.rec_expr.add(node);
        let dim1_id = self.add_or_get_val(dim1);
        let dim2_id = self.add_or_get_val(dim2);
        let dim3_id = self.add_or_get_val(dim3);
        let dim4_id = self.add_or_get_val(dim4);

        let new_node = Mdl::Input([name_id, dim1_id, dim2_id, dim3_id, dim4_id]);
        self.rec_expr.add(new_node)
    }

    pub fn conv2d(
        &mut self,
        inpt: Id,
        wght: Id,
        stride_h: i32,
        stride_w: i32,
        padding: i32,
        activation: i32,
    ) -> Id {
        let stride_h_id = self.add_or_get_val(stride_h);
        let stride_w_id = self.add_or_get_val(stride_w);
        let padding_id = self.add_or_get_val(padding);
        let activation_id = self.add_or_get_val(activation);

        let new_node = Mdl::Conv2d([
            stride_h_id,
            stride_w_id,
            padding_id,
            activation_id,
            inpt,
            wght,
        ]);
        self.rec_expr.add(new_node)
    }

    pub fn relu(&mut self, inpt: Id) -> Id {
        let new_node = Mdl::Relu(inpt);
        self.rec_expr.add(new_node)
    }

    pub fn add(&mut self, inpt_1: Id, inpt_2: Id) -> Id {
        let new_node = Mdl::Ewadd([inpt_1, inpt_2]);
        self.rec_expr.add(new_node)
    }

    pub fn matmul(&mut self, inpt_1: Id, inpt_2: Id) -> Id {
        let activation = ACTNONE;
        let act_id = self._get_or_add_val(activation);

        let new_node = Mdl::Matmul([act_id, inpt_1, inpt_2]);
        self.rec_expr.add(new_node)
    }

    pub fn mul(&mut self, inpt_1: Id, inpt_2: Id) -> Id {
        let new_node = Mdl::Ewmul([inpt_1, inpt_2]);
        self.rec_expr.add(new_node)
    }

    pub fn concat(&mut self, axis: i32, ndim: i32, inpt_1: Id, inpt_2: Id) -> Id {
        // Only support concat of 2 inputs for now
        // To support more, pass in a slice and create more concat nodes here
        let axis_id = self._get_or_add_val(axis);
        let ndim_id = self._get_or_add_val(ndim);

        let new_node = Mdl::Concat([axis_id, ndim_id, inpt_1, inpt_2]);
        self.rec_expr.add(new_node)
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
}

/// Struct for generating new names for weight tensors in the model
///
/// Generates names like w1, w2...
#[derive(Default)]
pub struct NameGen {
    count: i32,
}

impl NameGen {
    pub fn new_name(&mut self) -> String {
        let name = format!("w{}", self.count);
        self.count += 1;
        name
    }
}
