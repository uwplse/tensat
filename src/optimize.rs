use crate::{model::*, rewrites::*};
use egg::*;
use std::convert::TryInto;
use root::taso::*;
use std::time::{Duration, Instant};


/// Custom struct implementing our cost function
///
/// # Fields
/// 
/// - `egraph`: egraph, for getting metadata
pub struct TensorCost<'a> {
    pub egraph: &'a EGraph<Mdl, TensorAnalysis>
}

impl CostFunction<Mdl> for TensorCost<'_> {
    type Cost = f32;
    /// Getting total cost for the subtree rooted at enode. See egg::CostFunction
    /// trait for more information on interface.
    fn cost<C: FnMut(Id) -> Self::Cost>(&mut self, enode: &Mdl, mut costs: C) -> Self::Cost {
        let self_cost = get_self_cost(self.egraph, enode);
        enode.fold(self_cost, |sum, id| sum + costs(id))
    }
}


/// Gets cost for the enode itself.
///
/// This function gets the cost by calling TASO's get_or_create_{some_op}() 
/// functions with the tensor information stored in metadata. TASO side stores
/// hashmaps for OpBase objects. So here TASO side will simply lookup previously
/// created ops (with previously measured runtime). 
///
/// # Parameters
///
/// - `egraph`: E-graph of interest
/// - `enode`: enode to get cost for
///
/// # Returns
///
/// Cost for this enode.
fn get_self_cost(egraph: &EGraph<Mdl, TensorAnalysis>, enode: &Mdl) -> f32 {
    let x = |i: &Id| &egraph[*i].data;
    let mut g = egraph.analysis.graph.borrow_mut();
    match enode {
        Mdl::Num(_) | Mdl::Var(_) | Mdl::Inpt(_) => 0.0,

        Mdl::Relu(_a) => {
            // Check types
            let a_t_data = x(_a);
            assert!(a_t_data.dtype == DataKind::Tnsr);

            unsafe {
                // Get op
                let op = (*g.model).get_or_create_activation(*a_t_data.meta, OpType_OP_RELU, true);
                assert!(op != Op_INVALID_OP);
                (*op.ptr).runtime.clone()
            }
        },

        Mdl::Conv2d([_stride_h, _stride_w, _pad, _act, _inpt, _wght]) => {
            // Check types
            let _stride_h_data = x(_stride_h);
            let _stride_w_data = x(_stride_w);
            let _pad_data = x(_pad);
            let _act_data = x(_act);
            let _inpt_data = x(_inpt);
            let _wght_data = x(_wght);
            assert!(_stride_h_data.dtype == DataKind::Scalar);
            assert!(_stride_w_data.dtype == DataKind::Scalar);
            assert!(_pad_data.dtype == DataKind::Scalar);
            assert!(_act_data.dtype == DataKind::Scalar);
            assert!(_inpt_data.dtype == DataKind::Tnsr);
            assert!(_wght_data.dtype == DataKind::Tnsr);

            unsafe {
                // Get arguments
                let t_inpt = *_inpt_data.meta;
                let t_wght = *_wght_data.meta;
                let strideH = _stride_h_data.val;
                let strideW = _stride_w_data.val;
                let padding: PaddingMode = _pad_data.val.try_into().unwrap();
                let activation: ActiMode = _act_data.val.try_into().unwrap();

                // Get op
                //let start_time = Instant::now();
                let op = (*g.model).get_or_create_conv2d(t_inpt, t_wght, strideH, strideW, padding, activation);
                //let duration = start_time.elapsed();
                //println!("  Time taken getc conv: {:?}", duration);
                assert!(op != Op_INVALID_OP);
                (*op.ptr).runtime.clone()
            }
        },


        Mdl::Ewadd([_a, _b]) => {
            // Check types
            let _a_data = x(_a);
            let _b_data = x(_b);
            assert!(_a_data.dtype == DataKind::Tnsr);
            assert!(_b_data.dtype == DataKind::Tnsr);

            unsafe {
                // Get arguments
                let t_a = _a_data.meta;
                let t_b = _b_data.meta;
            
                // Get op
                //let start_time = Instant::now();
                let op = (*g.model).get_or_create_element(OpType_OP_EW_ADD, t_a, t_b);
                //let duration = start_time.elapsed();
                //println!("  Time taken getc ele: {:?}", duration);
                assert!(op != Op_INVALID_OP);
                (*op.ptr).runtime.clone()
            }
        },

        other => {
            println!("Get cost not implemented for: {:?}", other);
            0.0
        },
    }
}

/*
struct Cost;
impl CostFunction<Mdl> for Cost {
    type Cost = (f64, Vec<usize>);
    fn cost<C: FnMut(Id) -> Self::Cost>(&mut self, enode: &Mdl, mut costs: C) -> Self::Cost {
        let children_sizes = enode.fold(vec![], |mut sizes, id| {
            sizes.push(costs(id).1);
            sizes
        });
        layouts(enode)
            .into_iter()
            .map(|layout| Self::run_time(enode, layout, &children_sizes))
            .min_by(|(x, _), (y, _)| x.partial_cmp(y).unwrap())
            .unwrap()
        // TODO gotta calc output sizes
    }
}

struct Layout;
fn layouts(_e: &Mdl) -> Vec<Layout> {
    todo!()
}

impl Cost {
    fn run_time(
        _e: &Mdl,
        _layout: Layout,
        _sizes: &[Vec<usize>],
    ) -> <Cost as CostFunction<Mdl>>::Cost {
        todo!()
    }
}
*/