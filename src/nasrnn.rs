use crate::{input::*, model::*};
use egg::*;

const LENGTH: i32 = 5;
const HIDDEN_SIZE: i32 = 512;

fn combine(graph: &mut GraphConverter, x: Id, h: Id) -> Id {
    let w1 = graph.new_weight(vec![HIDDEN_SIZE, HIDDEN_SIZE]);
    let w2 = graph.new_weight(vec![HIDDEN_SIZE, HIDDEN_SIZE]);
    let t1 = graph.matmul(x, w1);
    let t2 = graph.matmul(h, w2);
    graph.add(t1, t2)
}

fn nas_node(graph: &mut GraphConverter, input: Id, x: Id) -> Id {
    let mut tmp = Vec::new();
    for i in 0..8 {
        tmp.push(combine(graph, x, input));
    }
    let mut midt = Vec::new();
    let t1 = graph.relu(tmp[0]);
    let t2 = graph.sigmoid(tmp[3]);
    midt.push(graph.add(t1, t2));
    let t1 = graph.sigmoid(tmp[1]);
    let t2 = graph.tanh(tmp[2]);
    midt.push(graph.add(t1, t2));
    let t1 = graph.sigmoid(tmp[4]);
    let t2 = graph.tanh(tmp[5]);
    midt.push(graph.mul(t1, t2));
    let t1 = graph.sigmoid(tmp[6]);
    let t2 = graph.relu(tmp[7]);
    midt.push(graph.mul(t1, t2));
    let t1 = graph.sigmoid(midt[1]);
    let t2 = graph.tanh(midt[2]);
    midt.push(graph.add(t1, t2));
    let t1 = graph.tanh(midt[0]);
    let t2 = graph.tanh(midt[3]);
    midt.push(graph.mul(t1, t2));
    let t1 = graph.tanh(midt[4]);
    let t2 = graph.tanh(midt[5]);
    midt.push(graph.mul(t1, t2));
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
    for x in xs {
        state = nas_node(&mut graph, state, x)
    }

    // Step 3: get the RexExpr
    graph.rec_expr()
}