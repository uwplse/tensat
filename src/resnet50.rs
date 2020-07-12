use crate::{model::*, input::*};
use egg::*;

#[derive(Default)]
struct NameGen {
    count: i32,
}

impl NameGen {
    fn get_name(&self) -> String {
        let name = format!("w{}", self.count);
        self.count += 1;
        name
    }
}

fn resnet_block(graph: &mut GraphConverter, input: Id, strides: (i32, i32), out_channels: i32, input_dim_1: i32, name_gen: &NameGen) -> Id {
    let w1 = graph.new_input(name_gen.get_name(), out_channels, input_dim_1, 1, 1);
    let t = graph.conv2d(input, w1, 1, 1, PSAME, ACTRELU);
    let w2 = graph.new_input(name_gen.get_name(), out_channels, out_channels, 3, 3);
    let t = graph.conv2d(t, w2, strides.0, strides.1, PSAME, ACTRELU);
    let w3 = graph.new_input(name_gen.get_name(), out_channels*4, out_channels, 1, 1);
    let t = graph.conv2d(t, w3, 1, 1, PSAME, ACTNONE);
    if (strides.0 > 1) || (input_dim_1 != out_channels*4) {
        let w4 = graph.new_input(name_gen.get_name(), out_channels*4, input_dim_1, 1, 1);
        let input = graph.conv2d(input, w4, strides.0, strides.1, PSAME, ACTRELU);
    }
    graph.relu(graph.add(input, t))
}

pub fn get_resnet50() -> RecExpr<Mdl> {
    let mut graph = GraphConverter::default();
    let name_gen = NameGen::default();

    let input = graph.new_input("input_1", 1, 64, 56, 56);
    let t = input;
    let mut input_dim_1 = 64;
    for i in 0..3 {
        let out_channels = 64;
        t = resnet_block(&graph, t, (1,1), out_channels, input_dim_1, &name_gen);
        input_dim_1 = out_channels * 4;
    }
    let mut strides = (2,2);
    for i in 0..4 {
        let out_channels = 128;
        t = resnet_block(&graph, t, strides, out_channels, input_dim_1, &name_gen);
        input_dim_1 = out_channels * 4;
        strides = (1,1);
    }
    strides = (2,2);
    for i in 0..6 {
        let out_channels = 256;
        t = resnet_block(&graph, t, strides, out_channels, input_dim_1, &name_gen);
        input_dim_1 = out_channels * 4;
        strides = (1,1);
    }
    strides = (2,2);
    for i in 0..3 {
        let out_channels = 512;
        t = resnet_block(&graph, t, strides, out_channels, input_dim_1, &name_gen);
        input_dim_1 = out_channels * 4;
        strides = (1,1);
    }

    graph.get_rec_expr()
}