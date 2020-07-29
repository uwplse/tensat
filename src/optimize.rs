use crate::{model::*, rewrites::*};
use egg::*;
use root::taso::*;
use std::convert::TryInto;
use std::time::{Duration, Instant};

/// Custom struct implementing our cost function
pub struct TensorCost<'a> {
    pub egraph: &'a EGraph<Mdl, TensorAnalysis>,
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
        Mdl::Num(_) | Mdl::Var(_) | Mdl::Input(_) | Mdl::Weight(_) | Mdl::Merge(_) | Mdl::Split0(_) | Mdl::Split1(_) => 0.0,

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
        }

        Mdl::Tanh(_a) => {
            // Check types
            let a_t_data = x(_a);
            assert!(a_t_data.dtype == DataKind::Tnsr);

            unsafe {
                // Get op
                let op = (*g.model).get_or_create_activation(*a_t_data.meta, OpType_OP_TANH, true);
                assert!(op != Op_INVALID_OP);
                (*op.ptr).runtime.clone()
            }
        }

        Mdl::Sigmoid(_a) => {
            // Check types
            let a_t_data = x(_a);
            assert!(a_t_data.dtype == DataKind::Tnsr);

            unsafe {
                // Get op
                let op =
                    (*g.model).get_or_create_activation(*a_t_data.meta, OpType_OP_SIGMOID, true);
                assert!(op != Op_INVALID_OP);
                (*op.ptr).runtime.clone()
            }
        }

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

            // Get arguments
            let stride_h = _stride_h_data.val;
            let stride_w = _stride_w_data.val;
            let padding: PaddingMode = _pad_data.val.try_into().unwrap();
            let activation: ActiMode = _act_data.val.try_into().unwrap();
            unsafe {
                let t_inpt = *_inpt_data.meta;
                let t_wght = *_wght_data.meta;
                // Get op
                let op = (*g.model)
                    .get_or_create_conv2d(t_inpt, t_wght, stride_h, stride_w, padding, activation);
                assert!(op != Op_INVALID_OP);
                (*op.ptr).runtime.clone()
            }
        }

        Mdl::Ewadd([_a, _b]) => {
            // Check types
            let _a_data = x(_a);
            let _b_data = x(_b);
            assert!(_a_data.dtype == DataKind::Tnsr);
            assert!(_b_data.dtype == DataKind::Tnsr);

            // Get arguments
            let t_a = _a_data.meta;
            let t_b = _b_data.meta;
            unsafe {
                // Get op
                let op = (*g.model).get_or_create_element(OpType_OP_EW_ADD, t_a, t_b);
                assert!(op != Op_INVALID_OP);
                (*op.ptr).runtime.clone()
            }
        }

        Mdl::Ewmul([_a, _b]) => {
            // Check types
            let _a_data = x(_a);
            let _b_data = x(_b);
            assert!(_a_data.dtype == DataKind::Tnsr);
            assert!(_b_data.dtype == DataKind::Tnsr);

            // Get arguments
            let t_a = _a_data.meta;
            let t_b = _b_data.meta;
            unsafe {
                // Get op
                let op = (*g.model).get_or_create_element(OpType_OP_EW_MUL, t_a, t_b);
                assert!(op != Op_INVALID_OP);
                (*op.ptr).runtime.clone()
            }
        }

        Mdl::Matmul([_act, _a, _b]) => {
            // Check types
            let _act_data = x(_act);
            let _a_data = x(_a);
            let _b_data = x(_b);
            assert!(_act_data.dtype == DataKind::Scalar);
            assert!(_a_data.dtype == DataKind::Tnsr);
            assert!(_b_data.dtype == DataKind::Tnsr);

            // Get arguments
            let activation: ActiMode = _act_data.val.try_into().unwrap();
            unsafe {
                let t_a = *_a_data.meta;
                let t_b = *_b_data.meta;
                // Get op
                let op = (*g.model).get_or_create_matmul(t_a, t_b, activation);
                assert!(op != Op_INVALID_OP);
                (*op.ptr).runtime.clone()
            }
        }

        Mdl::Concat([_axis, _ndim, _a, _b]) => {
            // Check types
            let _axis_data = x(_axis);
            let _ndim_data = x(_ndim);
            let _a_data = x(_a);
            let _b_data = x(_b);
            assert!(_axis_data.dtype == DataKind::Scalar);
            assert!(_ndim_data.dtype == DataKind::Scalar);
            assert!(_a_data.dtype == DataKind::Tnsr);
            assert!(_b_data.dtype == DataKind::Tnsr);

            // Get arguments
            let axis = _axis_data.val;
            let ndim = _ndim_data.val;
            unsafe {
                let t_a = *_a_data.meta;
                let t_b = *_b_data.meta;

                // Pass ownership to C++
                let mut inputs = vec![t_a, t_b];
                inputs.shrink_to_fit();
                assert!(inputs.len() == inputs.capacity());
                let ptr = inputs.as_mut_ptr();
                std::mem::forget(inputs);

                // Get op
                let mut need_copy = [false, false];
                let op = (*g.model).get_or_create_concat(axis, 2, ptr, need_copy.as_mut_ptr());
                assert!(op != Op_INVALID_OP);
                (*op.ptr).runtime.clone()
            }
        }

        Mdl::Poolmax([_inpt, _kernel_h, _kernel_w, _stride_h, _stride_w, _pad, _act]) => {
            // Check types
            let _kernel_h_data = x(_kernel_h);
            let _kernel_w_data = x(_kernel_w);
            let _stride_h_data = x(_stride_h);
            let _stride_w_data = x(_stride_w);
            let _pad_data = x(_pad);
            let _act_data = x(_act);
            let _inpt_data = x(_inpt);
            assert!(_kernel_h_data.dtype == DataKind::Scalar);
            assert!(_kernel_w_data.dtype == DataKind::Scalar);
            assert!(_stride_h_data.dtype == DataKind::Scalar);
            assert!(_stride_w_data.dtype == DataKind::Scalar);
            assert!(_pad_data.dtype == DataKind::Scalar);
            assert!(_act_data.dtype == DataKind::Scalar);
            assert!(_inpt_data.dtype == DataKind::Tnsr);

            // Get arguments
            let kernel_h = _kernel_h_data.val;
            let kernel_w = _kernel_w_data.val;
            let stride_h = _stride_h_data.val;
            let stride_w = _stride_w_data.val;
            let padding: PaddingMode = _pad_data.val.try_into().unwrap();
            let activation: ActiMode = _act_data.val.try_into().unwrap();
            unsafe {
                let t_inpt = *_inpt_data.meta;
                let t_wght = t_inpt.clone(); // Just a placeholder, t_wght won't be used in get_or_create_pool2d here

                // Get op
                let op = (*g.model).get_or_create_pool2d(
                    t_inpt,
                    t_wght,
                    OpType_OP_POOL2D_MAX,
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    padding,
                    activation,
                );
                assert!(op != Op_INVALID_OP);
                (*op.ptr).runtime.clone()
            }
        }

        Mdl::Split([_axis, _inpt]) => {
            // Check types
            let _axis_data = x(_axis);
            let _inpt_data = x(_inpt);
            assert!(_axis_data.dtype == DataKind::Scalar);
            assert!(_inpt_data.dtype == DataKind::Tnsr);

            // Get arguments
            let t_inpt = _inpt_data.meta;
            let axis = _axis_data.val;
            unsafe {
                // Get op
                let op = (*g.model).get_or_create_split1(t_inpt, axis, 2);
                assert!(op != Op_INVALID_OP);
                (*op.ptr).runtime.clone()
            }
        }

        Mdl::Enlarge([_a, _b]) => {
            // Check types
            let _a_data = x(_a);
            let _b_data = x(_b);
            assert!(_a_data.dtype == DataKind::Tnsr);
            assert!(_b_data.dtype == DataKind::Tnsr);

            // Get arguments
            unsafe {
                let t_a = *_a_data.meta;
                let t_b = *_b_data.meta;
                // Get op
                let op = (*g.model).get_or_create_enlarge(t_a, t_b);
                assert!(op != Op_INVALID_OP);
                (*op.ptr).runtime.clone()
            }
        }

        other => {
            println!("Get cost not implemented for: {:?}", other);
            0.0
        }
    }
}
