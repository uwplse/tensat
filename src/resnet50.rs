use crate::{input::*, model::*};
use egg::*;

fn resnet_block(
    graph: &mut GraphConverter,
    mut input: Id,
    strides: (i32, i32),
    out_channels: i32,
    input_dim_1: i32,
) -> Id {
    let w1 = graph.new_weight(vec![out_channels, input_dim_1, 1, 1]);
    let tmp = graph.conv2d(
        input, w1, /*stride_h=*/ 1, /*stride_w=*/ 1, /*padding=*/ PSAME,
        /*activation=*/ ACTRELU,
    );
    let w2 = graph.new_weight(vec![out_channels, out_channels, 3, 3]);
    let tmp = graph.conv2d(
        tmp, w2, /*stride_h=*/ strides.0, /*stride_w=*/ strides.1,
        /*padding=*/ PSAME, /*activation=*/ ACTRELU,
    );
    let w3 = graph.new_weight(vec![out_channels * 4, out_channels, 1, 1]);
    let tmp = graph.conv2d(
        tmp, w3, /*stride_h=*/ 1, /*stride_w=*/ 1, /*padding=*/ PSAME,
        /*activation=*/ ACTNONE,
    );
    if (strides.0 > 1) || (input_dim_1 != out_channels * 4) {
        let w4 = graph.new_weight(vec![out_channels * 4, input_dim_1, 1, 1]);
        input = graph.conv2d(
            input, w4, /*stride_h=*/ strides.0, /*stride_w=*/ strides.1,
            /*padding=*/ PSAME, /*activation=*/ ACTRELU,
        );
    }
    let tmp = graph.add(input, tmp);
    graph.relu(tmp)
}

/// Gets the RecExpr of a resnet50 model
pub fn get_resnet50() -> RecExpr<Mdl> {
    // Step 1: create a GraphConverter instance
    let mut graph = GraphConverter::default();

    // Step 2: define the graph
    let input = graph.new_input(vec![1, 64, 56, 56]);
    let mut tmp = input;
    let mut input_dim_1 = 64;
    for i in 0..3 {
        let out_channels = 64;
        tmp = resnet_block(
            &mut graph,
            tmp,
            /*strides=*/ (1, 1),
            out_channels,
            input_dim_1,
        );
        input_dim_1 = out_channels * 4;
    }

    let mut strides = (2, 2);
    for i in 0..4 {
        let out_channels = 128;
        tmp = resnet_block(&mut graph, tmp, strides, out_channels, input_dim_1);
        input_dim_1 = out_channels * 4;
        strides = (1, 1);
    }
    strides = (2, 2);
    for i in 0..6 {
        let out_channels = 256;
        tmp = resnet_block(&mut graph, tmp, strides, out_channels, input_dim_1);
        input_dim_1 = out_channels * 4;
        strides = (1, 1);
    }
    strides = (2, 2);
    for i in 0..3 {
        let out_channels = 512;
        tmp = resnet_block(&mut graph, tmp, strides, out_channels, input_dim_1);
        input_dim_1 = out_channels * 4;
        strides = (1, 1);
    }

    // Step 3: get the RexExpr
    graph.rec_expr()
}
