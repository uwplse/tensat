use crate::{input::*, model::*};
use egg::*;

fn benchnet_block(
    graph: &mut GraphConverter,
    mut input: Id,
    strides: (i32, i32),
    out_channels: i32,
    input_dim_1: i32,
) -> Id {
    let w1 = graph.new_weight(&[out_channels, input_dim_1, 1, 1]);
    let t = graph.conv2d(input, w1, 1, 1, PSAME, ACTRELU);
    let w2 = graph.new_weight(&[out_channels, out_channels, 3, 3]);
    let t = graph.conv2d(t, w2, strides.0, strides.1, PSAME, ACTRELU);
    let w3 = graph.new_weight(&[out_channels * 4, out_channels, 1, 1]);
    let t = graph.conv2d(t, w3, 1, 1, PSAME, ACTNONE);
    if (strides.0 > 1) || (input_dim_1 != out_channels * 4) {
        let w4 = graph.new_weight(&[out_channels * 4, input_dim_1, 1, 1]);
        input = graph.conv2d(input, w4, strides.0, strides.1, PSAME, ACTRELU);
    }
    let t = graph.add(input, t);
    let t = graph.mul(input, t);
    graph.relu(t)
}

/// Gets the RecExpr of a benchnet model
pub fn get_benchnet() -> RecExpr<Mdl> {
    // Step 1: create a GraphConverter instance
    let mut graph = GraphConverter::default();

    // Step 2: define the graph
    let input = graph.new_input(&[1, 64, 56, 56]);
    let mut t = input;
    let mut input_dim_1 = 64;
    let mut strides = (1, 1);
    for i in 0..3 {
        let out_channels = 64;
        t = benchnet_block(&mut graph, t, strides, out_channels, input_dim_1);
        input_dim_1 = out_channels * 4;
    }

    strides = (2, 2);
    for i in 0..4 {
        let out_channels = 128;
        t = benchnet_block(&mut graph, t, strides, out_channels, input_dim_1);
        input_dim_1 = out_channels * 4;
        strides = (1, 1);
    }
    strides = (2, 2);
    for i in 0..6 {
        let out_channels = 256;
        t = benchnet_block(&mut graph, t, strides, out_channels, input_dim_1);
        input_dim_1 = out_channels * 4;
        strides = (1, 1);
    }
    strides = (2, 2);
    for i in 0..3 {
        let out_channels = 512;
        t = benchnet_block(&mut graph, t, strides, out_channels, input_dim_1);
        input_dim_1 = out_channels * 4;
        strides = (1, 1);
    }

    // Step 3: get the RexExpr
    graph.rec_expr()
}
