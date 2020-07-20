use crate::{input::*, model::*};
use egg::*;

fn resnet_block(
    graph: &mut GraphConverter,
    mut input: Id,
    strides: (i32, i32),
    out_channels: i32,
    input_dim_1: i32,
) -> Id {
    let w1 = graph.new_weight(vec![out_channels * 4, input_dim_1, 1, 1]);
    let mut t = graph.conv2d(input, w1, 1, 1, PSAME, ACTRELU);

    let w4 = graph.new_weight(vec![out_channels, input_dim_1, 1, 1]);
    t = graph.conv2d(t, w4, strides.0, strides.1, PSAME, ACTRELU);

    t = graph.add(input, t);
    t = graph.mul(input, t);
    graph.relu(t)
}

/// Gets the RecExpr of a resnet50 model
pub fn get_testnet() -> RecExpr<Mdl> {
    // Step 1: create a GraphConverter instance, and a NameGen to generate new names
    let mut graph = GraphConverter::default();

    // Step 2: define the graph, in a TF/Pytorch like style
    let input = graph.new_input(vec![1, 64, 56, 56]);
    let mut t = input;
    let mut input_dim_1 = 64;
    let out_channels = 64;
    t = resnet_block(&mut graph, t, (1, 1), out_channels, input_dim_1);

    // Step 3: get the RexExpr
    graph.rec_expr()
}
