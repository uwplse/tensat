use crate::{model::*, input::*};
use egg::*;


fn resnet_block(graph: &mut GraphConverter, mut input: Id, strides: (i32, i32), out_channels: i32, input_dim_1: i32, name_gen: &mut NameGen) -> Id {
    let w1 = graph.new_input(&name_gen.get_name(), out_channels, input_dim_1, 1, 1);
    let t = graph.conv2d(input, w1, 1, 1, PSAME, ACTRELU);
    let w2 = graph.new_input(&name_gen.get_name(), out_channels, out_channels, 3, 3);
    let t = graph.conv2d(t, w2, strides.0, strides.1, PSAME, ACTRELU);
    let w3 = graph.new_input(&name_gen.get_name(), out_channels*4, out_channels, 1, 1);
    let t = graph.conv2d(t, w3, 1, 1, PSAME, ACTNONE);
    if (strides.0 > 1) || (input_dim_1 != out_channels*4) {
        let w4 = graph.new_input(&name_gen.get_name(), out_channels*4, input_dim_1, 1, 1);
        input = graph.conv2d(input, w4, strides.0, strides.1, PSAME, ACTRELU);
    }
    let t = graph.add(input, t);
    graph.relu(t)
}

/// Gets the RecExpr of a resnet50 model
pub fn get_resnet50() -> RecExpr<Mdl> {
    // Step 1: create a GraphConverter instance, and a NameGen to generate new names
    let mut graph = GraphConverter::default();
    let mut name_gen = NameGen::default();

    // Step 2: define the graph, in a TF/Pytorch like style
    let input = graph.new_input("input_1", 1, 64, 56, 56);
    let mut t = input;
    let mut input_dim_1 = 64;
    for i in 0..3 {
        let out_channels = 64;
        t = resnet_block(&mut graph, t, (1,1), out_channels, input_dim_1, &mut name_gen);
        input_dim_1 = out_channels * 4;
    }
    
    let mut strides = (2,2);
    for i in 0..4 {
        let out_channels = 128;
        t = resnet_block(&mut graph, t, strides, out_channels, input_dim_1, &mut name_gen);
        input_dim_1 = out_channels * 4;
        strides = (1,1);
    }
    strides = (2,2);
    for i in 0..6 {
        let out_channels = 256;
        t = resnet_block(&mut graph, t, strides, out_channels, input_dim_1, &mut name_gen);
        input_dim_1 = out_channels * 4;
        strides = (1,1);
    }
    strides = (2,2);
    for i in 0..3 {
        let out_channels = 512;
        t = resnet_block(&mut graph, t, strides, out_channels, input_dim_1, &mut name_gen);
        input_dim_1 = out_channels * 4;
        strides = (1,1);
    }

    // Step 3: get the RexExpr
    graph.get_rec_expr()
}