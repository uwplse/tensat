use crate::{input::*, model::*};
use egg::*;

/// Gets the RecExpr of a resnet50 model
pub fn get_testnet() -> RecExpr<Mdl> {
    // Step 1: create a GraphConverter instance, and a NameGen to generate new names
    let mut graph = GraphConverter::default();

    if true {
        let input = graph.new_input(&[32, 24, 24]);
        let tmp = graph.reshape(input, &[64, 12, 24]);
        let tmp = graph.transpose(tmp, &[1,0,2], true);
    } else {
        // Step 2: define the graph, in a TF/Pytorch like style
        let input = graph.new_input(&[1, 64, 56, 56]);
        let w1 = graph.new_weight(&[12, 64, 3, 3]);
        let w2 = graph.new_weight(&[12, 64, 5, 5]);
        let tmp = graph.conv2d(
            input, w1, /*stride_h=*/ 1, /*stride_w=*/ 1, /*padding=*/ PSAME,
            /*activation=*/ ACTRELU,
        );
        let tmp2 = graph.conv2d(
            input, w2, /*stride_h=*/ 1, /*stride_w=*/ 1, /*padding=*/ PSAME,
            /*activation=*/ ACTRELU,
        );
        let tmp3 = graph.add(tmp, tmp2);
    }

    // Step 3: get the RexExpr
    graph.rec_expr()
}
