#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

//use rand::prelude::*;
use root::taso::*;
use std::convert::TryInto;
use std::time::{Duration, Instant};

use egg::*;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum DataKind {
  Name,
  Scalar,
  Tnsr,
}

impl Default for DataKind {
    fn default() -> Self {
        DataKind::Name
    }
}

define_language! {
    pub enum Mdl {
        "input"     = Inpt([Id; 5]),
        "ewadd"     = Ewadd([Id; 2]),
        "ewmul"     = Ewmul([Id; 2]),
        "smul"      = Smul([Id; 2]),
        "transpose" = Transpose(Id),
        "matmul"    = Matmul([Id; 2]),
        "conv2d"    = Conv2d([Id; 6]), // conv2d's weight tensor kernel size can not be even, it seems that TASO's output shape computation is incorrect for even kernal size (like 4x4)
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
  pub graph: std::cell::RefCell<Box<Graph>>
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
  pub dtype: DataKind,
  pub val: i32,
  pub meta: TensorHandle,
}

impl Default for ValTnsr {
  fn default() -> Self {
    ValTnsr { meta: std::ptr::null_mut(), ..Default::default()}
  }
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
        // Check types
        assert!(x(a).dtype == DataKind::Tnsr);
        assert!(x(b).dtype == DataKind::Tnsr);

        // Get arguments
        let t_a = x(a).meta;
        let t_b = x(b).meta;

        // Create tensorhandle and get metadata
        unsafe {
          let mm = g.matmul(t_a, t_b, ActiMode_AC_MODE_NONE);
          let r_cost = (*(*mm).op.ptr).runtime;
          println!("Cost of matmul is {}", r_cost);
          Self::Data {dtype : DataKind::Tnsr, val : 0, meta : mm}
        }
      },

      Mdl::Conv2d([stride_h, stride_w, pad, act, inpt, wght]) => {
        // Check types
        assert!(x(stride_h).dtype == DataKind::Scalar);
        assert!(x(stride_w).dtype == DataKind::Scalar);
        assert!(x(pad).dtype == DataKind::Scalar);
        assert!(x(act).dtype == DataKind::Scalar);
        assert!(x(inpt).dtype == DataKind::Tnsr);
        assert!(x(wght).dtype == DataKind::Tnsr);

        // Get arguments
        let t_inpt = x(inpt).meta;
        let t_wght = x(wght).meta;
        let strideH = x(stride_h).val;
        let strideW = x(stride_w).val;
        let padding: PaddingMode = x(pad).val.try_into().unwrap();
        let activation: ActiMode = x(act).val.try_into().unwrap();

        // Create tensorhandle and get metadata
        unsafe {
          //let start_time = Instant::now();
          let res = g.conv2d1(t_inpt, t_wght, strideH, strideW, padding, activation);
          //let duration = start_time.elapsed();
          //println!("  Time taken get conv: {:?}", duration); 
          let r_cost = (*(*res).op.ptr).runtime;
          println!("Cost of conv2d is {}", r_cost);
          Self::Data {dtype : DataKind::Tnsr, val : 0, meta : res}
        }
      },

      Mdl::Ewadd([a, b]) => {
        // Check types
        assert!(x(a).dtype == DataKind::Tnsr);
        assert!(x(b).dtype == DataKind::Tnsr);

        // Get arguments
        let t_a = x(a).meta;
        let t_b = x(b).meta;

        // Create tensorhandle and get metadata
        unsafe {
          //let start_time = Instant::now();
          let res = g.element(OpType_OP_EW_ADD, t_a, t_b);
          //let duration = start_time.elapsed();
          //println!("  Time taken get ele: {:?}", duration);
          let r_cost = (*(*res).op.ptr).runtime;
          println!("Cost of ewadd is {}", r_cost);
          Self::Data {dtype : DataKind::Tnsr, val : 0, meta : res}
        }
      },

      Mdl::Relu(a) => {
        assert!(x(a).dtype == DataKind::Tnsr);
        let t_a = x(a).meta;

        unsafe {
          let relu = g.relu(t_a, true);
          let r_cost = (*(*relu).op.ptr).runtime;
          println!("Cost of relu is {}", r_cost);
          Self::Data {dtype : DataKind::Tnsr, val : 0, meta : relu}
        }
      },
      
      Mdl::Inpt([name, dim1, dim2, dim3, dim4]) => {
        assert!(x(name).dtype == DataKind::Name);
        assert!(x(dim1).dtype == DataKind::Scalar);
        assert!(x(dim2).dtype == DataKind::Scalar);
        assert!(x(dim3).dtype == DataKind::Scalar);
        assert!(x(dim4).dtype == DataKind::Scalar);

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
          Self::Data {dtype : DataKind::Tnsr, val : 0, meta : inp}
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
        Self::Data { dtype : DataKind::Scalar, val : *_n, meta : std::ptr::null_mut() }
      },

      Mdl::Var(_s) => {
        Self::Data { dtype : DataKind::Name, val : 0, meta : std::ptr::null_mut() }
      },

      other => {println!("{:?}", other); todo!()}
    }
  }
 
  // TODO may not need modify to do anything?
  fn modify(egraph: &mut EGraph<Mdl, Self>, id: Id) {
  }
}
