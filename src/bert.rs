use crate::{input::*, model::*};
use egg::*;

const SEQ_LENGTH: i32 = 64;
const HIDDEN_DIMS: i32 = 1024;


fn attention(graph: &mut GraphConverter, input: Id, heads: i32, input_dim_1: i32) -> (Id, Id) {
    let d_model = input_dim_1;
    let d_k = d_model / heads;
    assert!(input_dim_1 % heads == 0);
    let mut weights = Vec::new();
    for i in 0..3 {
        weights.push(graph.new_weight(&[d_model, d_model]));
    }
    // compute query, key, value tensors
    let q = graph.matmul(input, weights[0]);
    let k = graph.matmul(input, weights[1]);
    let v = graph.matmul(input, weights[2]);
    // reshape query, key, value to multiple heads
    let q = graph.reshape(q, &[64, 16, 64]);
    let k = graph.reshape(k, &[64, 16, 64]);
    let v = graph.reshape(v, &[64, 16, 64]);
    // transpose query, key, value for batched matmul
    let q = graph.transpose(q, &[1,0,2], true);
    let k = graph.transpose(k, &[1,0,2], true);
    let v = graph.transpose(v, &[1,0,2], true);
    // perform matrix multiplications
    let logits = graph.matmul(q, k);
    let output = graph.matmul(logits, v);
    // transpose the output back
    let output = graph.transpose(output,&[1,0,2], true);
    let output = graph.reshape(output, &[64, 1024]);

    // a final linear layer
    let linear = graph.new_weight(&[d_model, d_model]);
    let next_in = graph.matmul(input, linear);
    (next_in, output)
}

/// Gets the RecExpr of a bert model
pub fn get_bert() -> RecExpr<Mdl> {
    // Step 1: create a GraphConverter instance
    let mut graph = GraphConverter::default();

    // Step 2: define the graph
    let input = graph.new_input(&[SEQ_LENGTH, HIDDEN_DIMS]);
    let input = graph.relu(input);
    let mut tmp = input;
    let mut current: Id = Default::default();
    for i in 0..8 {
        let (next_in, output) = attention(&mut graph, tmp, 16, HIDDEN_DIMS);
        tmp = next_in;
        if i == 0 {
            current = output;
        } else {
            current = graph.concat(0, 2, current, output);
        }
    }

    // Step 3: get the RexExpr
    graph.rec_expr()
}
