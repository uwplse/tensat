use crate::{model::*, input::*};
use egg::*;


pub fn get_resnet50() -> RecExpr<Mdl> {
    let mut graph = GraphConverter::default();

    let id = graph.new_input("w1", 3,3,3,3);

    graph.get_rec_expr()
}