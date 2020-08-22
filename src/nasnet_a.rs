use crate::{input::*, model::*};
use egg::*;

/// input.dim(1)
fn squeeze(graph: &mut GraphConverter, out_channels: i32, input: Id, input_dim_1: i32) -> Id {
    let weight = graph.new_weight(&[out_channels, input_dim_1, 1, 1]);
    graph.conv2d(input, weight, /*stride_h=*/ 1, /*stride_w=*/ 1,
        /*padding=*/ PSAME, /*activation=*/ ACTRELU);
}

fn fit(graph: &mut GraphConverter, current: Id, input: Id, cur_dim_1: i32, cur_dim_2: i32, input_dim_1: i32, input_dim_2: i32) -> Id {
    if input_dim_2 == cur_dim_2 {
        squeeze(graph, cur_dim_1, input, input_dim_1)
    } else {
        let weight = graph.new_weight(&[cur_dim_1, input_dim_1, 3, 3]);
        graph.conv2d(input, weight, /*stride_h=*/ 2, /*stride_w=*/ 2,
            /*padding=*/ PSAME, /*activation=*/ ACTRELU)
    }
}

/// cur_dim_1, prev_dim_1, cur_dim_2, prev_dim_2
fn normal_cell(graph: &mut GraphConverter, prev: Id, cur: Id, out_channels: i32, cur_dim_1: i32, prev_dim_1: i32) -> Id {
    let cur = squeeze(graph, out_channels, cur, cur_dim_1);
    let prev = fit(graph, cur, prev);
    let mut tmp = Vec::new();
    tmp.push(seperable_conv(graph, cur, out_channels, /*kernels=*/(3,3), /*strides=*/(1,1), /*padding=*/ PSAME));
}

/// Gets the RecExpr of a nasnet_a model
pub fn get_nasneta() -> RecExpr<Mdl> {
    // Step 1: create a GraphConverter instance
    let mut graph = GraphConverter::default();

    // Step 2: define the graph
    let mut input = graph.new_input(&[1,3,224,224]);
    let weight = graph.new_weight(&[64,3,7,7]);
    input = graph.conv2d(
        input, weight, /*stride_h=*/ 2, /*stride_w=*/ 2,
        /*padding=*/ PSAME, /*activation=*/ ACTRELU,
    );
    input = graph.maxpool2d(input, /*kernel_h=*/ 3, /*kernel_w=*/ 3, /*stride_h=*/ 2, /*stride_w=*/ 2,
        /*padding=*/ PSAME);

    let mut out_channels = 128;
    for i in 0..3 {
        let mut prev = input;
        let mut cur = input;
        for j in 0..5 {
            let tmp = normal_cell(&mut graph, prev, cur, out_channels);
            prev = cur;
            cur = tmp;
        }
        out_channels = out_channels * 2;
        input = reduction_cell(&mut graph, prev, cur, out_channels);
    }

    // Step 3: get the RexExpr
    graph.rec_expr()
}
