use crate::{input::*, model::*};
use egg::*;


fn squeeze(graph: &mut GraphConverter, out_channels: i32, input: TensorInfo) -> TensorInfo {
    let weight = graph.new_weight(&[out_channels, input.shape[1], 1, 1]);
    graph.conv2d(input, weight, /*stride_h=*/ 1, /*stride_w=*/ 1,
        /*padding=*/ PSAME, /*activation=*/ ACTRELU)
}

fn fit(graph: &mut GraphConverter, current: TensorInfo, input: TensorInfo) -> TensorInfo {
    if input.shape[2] == current.shape[2] {
        squeeze(graph, current.shape[1], input)
    } else {
        let weight = graph.new_weight(&[current.shape[1], input.shape[1], 3, 3]);
        graph.conv2d(input, weight, /*stride_h=*/ 2, /*stride_w=*/ 2,
            /*padding=*/ PSAME, /*activation=*/ ACTRELU)
    }
}

fn seperable_conv(graph: &mut GraphConverter, input: TensorInfo, out_channels: i32, kernels: (i32, i32), strides: (i32, i32), padding: i32) -> TensorInfo {
    assert!(input.shape[1] % out_channels == 0);
    let weight_1 = graph.new_weight(&[out_channels, input.shape[1] / out_channels, kernels.0, kernels.1]);
    let tmp = graph.conv2d(input, weight_1, /*stride_h=*/ strides.0, /*stride_w=*/ strides.1, /*padding=*/ padding, /*activation=*/ ACTNONE);
    let weight_2 = graph.new_weight(&[out_channels, tmp.shape[1], 1, 1]);
    graph.conv2d(tmp, weight_2, /*stride_h=*/ 1, /*stride_w=*/ 1,
        /*padding=*/ PSAME, /*activation=*/ ACTNONE)
}

fn normal_cell(graph: &mut GraphConverter, prev: TensorInfo, cur: TensorInfo, out_channels: i32) -> TensorInfo {
    let cur = squeeze(graph, out_channels, cur);
    let prev = fit(graph, cur, prev);
    let mut tmp = Vec::new();
    tmp.push(seperable_conv(graph, cur, out_channels, /*kernels=*/(3,3), /*strides=*/(1,1), /*padding=*/ PSAME));
    tmp.push(cur);
    tmp.push(seperable_conv(graph, prev, out_channels, /*kernels=*/(3,3), /*strides=*/(1,1), /*padding=*/ PSAME));
    tmp.push(seperable_conv(graph, cur, out_channels, /*kernels=*/(3,3), /*strides=*/(1,1), /*padding=*/ PSAME));
    tmp.push(graph.avgpool2d(cur, /*kernel_h=*/ 3, /*kernel_w=*/ 3, /*stride_h=*/ 1, /*stride_w=*/ 1, /*padding=*/ PSAME));
    tmp.push(prev);
    tmp.push(graph.avgpool2d(prev, /*kernel_h=*/ 3, /*kernel_w=*/ 3, /*stride_h=*/ 1, /*stride_w=*/ 1, /*padding=*/ PSAME));
    tmp.push(graph.avgpool2d(prev, /*kernel_h=*/ 3, /*kernel_w=*/ 3, /*stride_h=*/ 1, /*stride_w=*/ 1, /*padding=*/ PSAME));
    tmp.push(seperable_conv(graph, prev, out_channels, /*kernels=*/(3,3), /*strides=*/(1,1), /*padding=*/ PSAME));
    tmp.push(seperable_conv(graph, prev, out_channels, /*kernels=*/(3,3), /*strides=*/(1,1), /*padding=*/ PSAME));
    assert!(tmp.len() == 10);
    let mut outputs = Vec::new();
    for i in 0..5 {
        outputs.push(graph.add(tmp[2*i], tmp[2*i+1]));
    }

    graph.concat_multi(/*axis=*/1, outputs[0].n_dim as i32, &outputs)
    /*let mut combined = outputs[0];
    for i in 1..outputs.len() {
        combined = graph.concat(/*axis=*/1, combined.n_dim as i32, combined, outputs[i])
    }
    combined*/
}

fn reduction_cell(graph: &mut GraphConverter, prev: TensorInfo, cur: TensorInfo, out_channels: i32) -> TensorInfo {
    let cur = squeeze(graph, out_channels, cur);
    let prev = fit(graph, cur, prev);
    let mut tmp = Vec::new();
    let mut outputs = Vec::new();
    tmp.push(seperable_conv(graph, prev, out_channels, /*kernels=*/(7,7), /*strides=*/(2,2), /*padding=*/ PSAME));
    tmp.push(seperable_conv(graph, cur, out_channels, /*kernels=*/(5,5), /*strides=*/(2,2), /*padding=*/ PSAME));
    outputs.push(graph.add(tmp[0], tmp[1]));
    tmp.push(graph.maxpool2d(cur, /*kernel_h=*/ 3, /*kernel_w=*/ 3, /*stride_h=*/ 2, /*stride_w=*/ 2, /*padding=*/ PSAME));
    tmp.push(seperable_conv(graph, prev, out_channels, /*kernels=*/(7,7), /*strides=*/(2,2), /*padding=*/ PSAME));
    outputs.push(graph.add(tmp[2], tmp[3]));
    tmp.push(graph.avgpool2d(cur, /*kernel_h=*/ 3, /*kernel_w=*/ 3, /*stride_h=*/ 2, /*stride_w=*/ 2, /*padding=*/ PSAME));
    tmp.push(seperable_conv(graph, prev, out_channels, /*kernels=*/(5,5), /*strides=*/(2,2), /*padding=*/ PSAME));
    outputs.push(graph.add(tmp[4], tmp[5]));
    tmp.push(graph.maxpool2d(cur, /*kernel_h=*/ 3, /*kernel_w=*/ 3, /*stride_h=*/ 2, /*stride_w=*/ 2, /*padding=*/ PSAME));
    tmp.push(seperable_conv(graph, outputs[0], out_channels, /*kernels=*/(3,3), /*strides=*/(1,1), /*padding=*/ PSAME));
    outputs.push(graph.add(tmp[6], tmp[7]));
    tmp.push(graph.avgpool2d(outputs[0], /*kernel_h=*/ 3, /*kernel_w=*/ 3, /*stride_h=*/ 1, /*stride_w=*/ 1, /*padding=*/ PSAME));
    tmp.push(outputs[1]);
    outputs.push(graph.add(tmp[8], tmp[9]));

    graph.concat_multi(/*axis=*/1, outputs[0].n_dim as i32, &outputs)
    /*let mut combined = outputs[0];
    for i in 1..outputs.len() {
        combined = graph.concat(/*axis=*/1, combined.n_dim as i32, combined, outputs[i])
    }
    combined*/
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
