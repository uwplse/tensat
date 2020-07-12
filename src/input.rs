use crate::model::*;
use egg::*;
use std::collections::HashMap;

#[derive(Default)]
pub struct GraphConverter {
    rec_expr: RecExpr<Mdl>,
    scalar_map: HashMap<i32, Id>,
}


impl GraphConverter {
    pub fn new_input(&mut self, name: &str, dim1: i32, dim2: i32, dim3: i32, dim4: i32) -> Id {
        let node = Mdl::Var(Symbol::from(name));
        let name_id = self.rec_expr.add(node);

        let dim1_id = self._get_or_add_val(dim1);
        let dim2_id = self._get_or_add_val(dim2);
        let dim3_id = self._get_or_add_val(dim3);
        let dim4_id = self._get_or_add_val(dim4);

        let input_node = Mdl::Input([name_id, dim1_id, dim2_id, dim3_id, dim4_id]);
        self.rec_expr.add(input_node)
    }

    fn _get_or_add_val(&mut self, val: i32) -> Id {
        match self.scalar_map.get(&val) {
            Some(id) => *id,
            None => {
                let node = Mdl::Num(val);
                let id = self.rec_expr.add(node);
                self.scalar_map.insert(val, id);
                id
            },
        }
    }

    pub fn get_rec_expr(&self) -> RecExpr<Mdl> {
        self.rec_expr.clone()
    }
}