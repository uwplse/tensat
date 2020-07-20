use crate::{input::*, model::*};
use egg::*;

const LENGTH: i32 = 5;
const HIDDEN_SIZE: i32 = 512;

fn combine(graph: &mut GraphConverter, x: Id, h: Id) -> Id {
    let w1 = graph.new_weight(vec![HIDDEN_SIZE, HIDDEN_SIZE]);
    let w2 = graph.new_weight(vec![HIDDEN_SIZE, HIDDEN_SIZE]);
    graph.add(graph.matmul(x, w1), graph.matmul(h, w2))
}

fn nas_node(graph: &mut GraphConverter, input: Id, x: Id) -> Id {
    let mut tmp = Vec::new();
    for i in 0..8 {
        tmp.push(combine(&mut graph, x, input));
    }
    let mut midt = Vec::new();
    midt.push(graph.add(graph.relu(tmp[0]), graph.sigmoid([tmp[3]])));
    midt.push(graph.add(graph.sigmoid(tmp[1]), graph.tanh(tmp[2])));
    midt.push(graph.mul(graph.sigmoid(tmp[4]), graph.tanh(tmp[5])));
    midt.push(graph.mul(graph.sigmoid(tmp[6]), graph.relu(tmp[7])));
    midt.push(graph.add(graph.sigmoid(midt[1]), graph.tanh(midt[2])));
    midt.push(graph.mul(graph.tanh(midt[0]), graph.tanh(midt[3])));
    midt.push(graph.mul(graph.tanh(midt[4]), graph.tanh(midt[5])));
    graph.tanh(midt[6])
}


/// Gets the RecExpr of a nasrnn model
pub fn get_nasrnn() -> RecExpr<Mdl> {
    // Step 1: create a GraphConverter instance
    let mut graph = GraphConverter::default();

    // Step 2: define the graph
    let mut xs = Vec::new();
    for i in 0..LENGTH {
        xs.push(graph.new_input(vec![1, HIDDEN_SIZE]));
    }
    let mut state = graph.new_weight(vec![1, HIDDEN_SIZE]);
    for i in 0..LENGTH {
        state = nas_node(&mut graph, state, xs[i])
    }

    // Step 3: get the RexExpr
    graph.rec_expr()
}