#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

use rand::prelude::*;
use root::taso::*;

use egg::*;

define_language! {
    pub enum Model {
        "ewadd"     = Ewadd([Id; 2]),
        "ewmul"     = Ewmul([Id; 2]),
        "smul"      = Smul([Id; 2]),
        "transpose" = Transpose(Id),
        "matmul"    = Matmul([Id; 2]),
        "conv2d"    = Conv2d([Id; 6]),
        "enlarge"   = Enlarge([Id; 3]),
        "relu"      = Relu(Id),
        "poolavg"   = Poolavg([Id; 6]),
        "poolmax"   = Poolmax([Id; 6]),
        "concat"    = Concat([Id; 3]),
        "split_0"   = Split0([Id; 2]),
        "split_1"   = Split1([Id; 2]),
        "Cpool"     = Cpool([Id; 2]),
        "Iconv"     = Iconv([Id; 2]),
        // Anone = 0
        // Arelu = 2
        // Psame = 0
        "Imatmul"   = Imatmul,
        "Iewmul"    = Iewmul,
        Num(i32),
        Var(Symbol),
    }
}

#[derive(Default)]
struct TensorAnalysis;

#[derive(Debug)]
struct Tnsr {
  cost: f32,
  graph: *mut Graph,
  meta: TensorHandle,
}

// unsafe fn new_w(graph: &mut Graph, mut v: Vec<i32>, rng: &mut impl Rng) -> *mut Tensor {
//       let vol = v.iter().product();
//       let data: Vec<f32> = (0..vol).map(|_| rng.gen()).collect();
//       graph.new_weight(v.len() as i32, v.as_ptr(), data.as_ptr())
// }
// 
// unsafe fn new_i(graph: &mut Graph, v: Vec<i32>) -> *mut Tensor {
//       graph.new_input(v.len() as i32, v.as_ptr())
// }

impl Analysis<Model> for TensorAnalysis {
  type Data = Tnsr;

  fn merge(&self, to: &mut Self::Data, from: Self::Data) -> bool {
    if to.cost > from.cost {
      *to = from;
      true
    } else { false }
  }

  fn make(egraph: &EGraph<Model, Self>, enode: &Model) -> Self::Data {
    let x = |i: &Id| &egraph[*i].data;
    match enode {
      Model::Matmul([a, b]) => {
          let g = x(a).graph;
          let t_a = x(a).meta;
          let t_b = x(b).meta;

          unsafe { // very unsafe sketchy af
            let mm = g.as_mut().unwrap().matmul(t_a, t_b, ActiMode_AC_MODE_NONE);
            Tnsr {cost : (*(*mm).op.ptr).runtime, graph : g, meta : mm}
          }
        },
      Model::Relu(a) => {
          let g = x(a).graph;
          let t_a = x(a).meta;

          unsafe { // very unsafe sketchy af
            let relu = g.as_mut().unwrap().relu(t_a, true);
            Tnsr {cost : (*(*relu).op.ptr).runtime, graph : g, meta : relu}
          }
        },
      Model::Concat([a, b, c]) => {
          let g = x(b).graph;
          // let t_a = x(a).meta;
          // TODO deal with non tensors
          let t_b = x(b).meta;
          let t_c = x(c).meta;

          unsafe { // very unsafe sketchy af
            let cat = g.as_mut().unwrap().concat(1, 2, vec![t_b, t_c].as_ptr());
            Tnsr {cost : (*(*cat).op.ptr).runtime, graph : g, meta : cat}
          }
        },
      // Model::Split_0([a, b, c]) => {
      //     let g = x(b).graph;
      //     // let t_a = x(a).meta;
      //     // TODO deal with non tensors
      //     let t_b = x(b).meta;
      //     let t_c = x(c).meta;

      //     unsafe { // very unsafe sketchy af
      //       let cat = g.as_mut().unwrap().split(1, 2, vec![t_b, t_c].as_ptr());
      //       Tnsr {cost : (*(*cat).op.ptr).runtime, graph : g, meta : cat}
      //     }
      //   },
      _ => todo!()
    }
  }
// unsafe fn new_i(graph: &mut Graph, v: Vec<i32>) -> *mut Tensor {
//       graph.new_input(v.len() as i32, v.as_ptr())
// }
 
  fn modify(egraph: &mut EGraph<Model, Self>, id: Id) {
    todo!()
  }
}
