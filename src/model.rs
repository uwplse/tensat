#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

use rand::prelude::*;
use root::taso::*;

use egg::*;

#[derive(Debug, PartialEq)]
enum DataType {
  Name,
  Scalar,
  Tnsr,
}

define_language! {
    pub enum Mdl {
        "input"     = Inpt([Id; 5]),
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
        // NOTE refer to TASO for the const values
        // Anone = 0
        // Arelu = 2
        // Psame = 0
        "Imatmul"   = Imatmul,
        "Iewmul"    = Iewmul,
        Num(i32),
        Var(Symbol),
    }
}

pub struct TensorAnalysis {
  graph: std::cell::RefCell<Box<Graph>>
}

impl Default for TensorAnalysis {
  fn default() -> Self {
    unsafe {
      // NOTE Box heap-allocates, otherwise any pointer from
      // C++ may be dangling
      let mut graph = Box::new(Graph::new());
      Graph_Graph(&mut *graph);
      TensorAnalysis { graph: std::cell::RefCell::new(graph) }
    } 
  } 
}

#[derive(Debug, Clone)]
pub struct ValTnsr {
  dtype: DataType,
  val: i32,
  meta: TensorHandle,
}

impl Analysis<Mdl> for TensorAnalysis {
  type Data = ValTnsr;

  fn merge(&self, to: &mut Self::Data, from: Self::Data) -> bool {
    /*if to.cost > from.cost {
      *to = from;
      true
    } else { false }*/
    *to = from;
    true
  }

  fn make(egraph: &EGraph<Mdl, Self>, enode: &Mdl) -> Self::Data {
    let x = |i: &Id| &egraph[*i].data;
    let mut g = egraph.analysis.graph.borrow_mut();
    match enode {
      Mdl::Matmul([a, b]) => {
        assert!(x(a).dtype == DataType::Tnsr);
        assert!(x(b).dtype == DataType::Tnsr);
        let t_a = x(a).meta;
        let t_b = x(b).meta;

        unsafe {
          let mm = g.matmul(t_a, t_b, ActiMode_AC_MODE_NONE);
          let r_cost = (*(*mm).op.ptr).runtime;
          println!("Cost of matmul is {}", r_cost);
          Self::Data {dtype : DataType::Tnsr, val : 0, meta : mm}
        }
      },
      Mdl::Relu(a) => {
        assert!(x(a).dtype == DataType::Tnsr);
        let t_a = x(a).meta;

        unsafe {
          let relu = g.relu(t_a, true);
          let r_cost = (*(*relu).op.ptr).runtime;
          println!("Cost of relu is {}", r_cost);
          Self::Data {dtype : DataType::Tnsr, val : 0, meta : relu}
        }
      },
      
      Mdl::Inpt([name, dim1, dim2, dim3, dim4]) => {
        assert!(x(name).dtype == DataType::Name);
        assert!(x(dim1).dtype == DataType::Scalar);
        assert!(x(dim2).dtype == DataType::Scalar);
        assert!(x(dim3).dtype == DataType::Scalar);
        assert!(x(dim4).dtype == DataType::Scalar);

        unsafe { // very unsafe sketchy
          // NOTE all this just to pass ownership
          // to C++, not sure if necessary
          let mut dims = vec![x(dim1).val, x(dim2).val, x(dim3).val, x(dim4).val];
          dims.shrink_to_fit();
          assert!(dims.len() == dims.capacity());
          let ptr = dims.as_mut_ptr();
          std::mem::forget(ptr);

          let inp = g.new_input(4, ptr);
          let r_cost = (*(*inp).op.ptr).runtime;
          println!("Cost of input is {}", r_cost);
          Self::Data {dtype : DataType::Tnsr, val : 0, meta : inp}
        }
      },
      //Mdl::Concat([a, b, c]) => {
      //    // let t_a = x(a).meta;
      //    let t_b = x(b).meta;
      //    let t_c = x(c).meta;

      //    unsafe { // very unsafe sketchy
      //      let cat = g.concat(1, 2, vec![t_b, t_c].as_ptr());
      //      Self::Data {cost : (*(*cat).op.ptr).runtime, meta : cat}
      //    }
      //  },
      
      Mdl::Num(_n) => {
        Self::Data { dtype : DataType::Scalar, val : *_n, meta : std::ptr::null_mut() }
      },

      Mdl::Var(_s) => {
        Self::Data { dtype : DataType::Name, val : 0, meta : std::ptr::null_mut() }
      },

      other => {println!("{:?}", other); todo!()}
    }
  }
 
  // TODO may not need modify to do anything?
  fn modify(egraph: &mut EGraph<Mdl, Self>, id: Id) {
  }
}
