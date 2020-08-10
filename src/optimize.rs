use crate::{model::*, rewrites::*};
use egg::*;
use root::taso::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::collections::HashSet;
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
        Mdl::Num(_)
        | Mdl::Var(_)
        | Mdl::Input(_)
        | Mdl::Weight(_)
        | Mdl::Merge(_)
        | Mdl::Split0(_)
        | Mdl::Split1(_)
        | Mdl::Reshape(_)
        | Mdl::Transpose(_) => 0.0,

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

/// Prepare the data for formulation ILP
///
/// # Returns
///
/// - `m_id_map`: list of EClass Id's each index m refers to
/// - `e_m`: each entry is the list of nodes i within eclass m
/// - `h_i`: each entry is the list of children EClass indices for node i
/// - `cost_i`: self cost for each node i
/// - `g_i`: which EClass index does node i belong to
/// - `root_m`: EClass index of the root eclass
/// - `i_to_nodes: Vector of enodes, ordered by index i
pub fn prep_ilp_data(
    egraph: &EGraph<Mdl, TensorAnalysis>,
    root: Id,
) -> (
    Vec<Id>,
    Vec<Vec<usize>>,
    Vec<Vec<usize>>,
    Vec<f32>,
    Vec<usize>,
    usize,
    Vec<Mdl>,
) {
    let m_id_map: Vec<Id> = egraph.classes().map(|c| egraph.find(c.id)).collect();
    assert!(m_id_map.len() == egraph.number_of_classes());
    let id_m_map: HashMap<Id, usize> = m_id_map
        .iter()
        .enumerate()
        .map(|(i, id)| (*id, i))
        .collect();

    let num_classes = egraph.number_of_classes();
    let num_nodes = egraph.total_size();
    let mut i_to_nodes: Vec<Mdl> = Vec::with_capacity(num_nodes);
    let mut e_m: Vec<Vec<usize>> = vec![Vec::new(); num_classes];
    let mut h_i: Vec<Vec<usize>> = Vec::with_capacity(num_nodes);
    let mut cost_i: Vec<f32> = Vec::with_capacity(num_nodes);
    let mut g_i: Vec<usize> = Vec::with_capacity(num_nodes);

    let mut i = 0;
    for class in egraph.classes() {
        let m = *id_m_map.get(&egraph.find(class.id)).unwrap();
        for node in class.iter() {
            i_to_nodes.push(node.clone());
            e_m[m].push(i);
            h_i.push(
                node.children()
                    .iter()
                    .map(|id| *id_m_map.get(&egraph.find(*id)).unwrap())
                    .collect(),
            );
            cost_i.push(get_self_cost(egraph, node));
            g_i.push(m);
            i += 1;
        }
    }

    let root_m = *id_m_map.get(&egraph.find(root)).unwrap();

    (m_id_map, e_m, h_i, cost_i, g_i, root_m, i_to_nodes)
}

/// Struct for storing the solved results from ILP
#[derive(Debug, Serialize, Deserialize)]
pub struct SolvedResults {
    /// The solved values for the variables associated with each node
    pub solved_x: Vec<i32>,
    /// The minimum total cost found
    pub cost: f32,
}

/// Construct the RecExpr of the optimized graph extracted
///
/// This function does the construction recursively with memoization. Call it with eclass=root
/// will construct the whole extracted graph
///
/// # Parameters
///
/// - `node_picked`: hashmap storing which node is picked for each EClass ID
/// - `eclass`: The EClass ID that we aim to construct as root
/// - `added_memo`: Map from EClass ID to RecExpr ID. Storing the eclasses that were already added
/// - `egraph`: E-graph of interest
/// - `expr`: the RecExpr storing the optimized graph, it is constructed within this function
///
/// # Returns
///
/// - The ID (index) in the output RecExpr for the eclass passed in as argument
pub fn construct_best_rec(
    node_picked: &HashMap<Id, Mdl>,
    eclass: Id,
    added_memo: &mut HashMap<Id, Id>,
    egraph: &EGraph<Mdl, TensorAnalysis>,
    expr: &mut RecExpr<Mdl>,
) -> Id {
    let id = egraph.find(eclass);

    match added_memo.get(&id) {
        Some(id_expr) => *id_expr,
        None => {
            let node = node_picked.get(&id).unwrap().clone().map_children(|child| {
                construct_best_rec(node_picked, child, added_memo, egraph, expr)
            });
            let id_expr = expr.add(node);
            assert!(added_memo.insert(id, id_expr).is_none());
            id_expr
        }
    }
}


pub fn get_init_solution(
    egraph: &EGraph<Mdl, TensorAnalysis>,
    root: Id,
    costs: &HashMap<Id, (f32, Mdl)>,
    g_i: &[usize],
    nodes_to_i: &HashMap<Mdl, usize>,
) -> (
    Vec<usize>,
    Vec<usize>,
) {
    let mut nodes: Vec<Mdl> = Vec::new();
    // added_memo maps eclass id to id in expr
    let mut added_memo: HashSet<Id> = Default::default();
    get_init_rec(egraph, root, &mut added_memo, costs, &mut nodes);

    let i_list: Vec<usize> = nodes.iter().map(|node| *nodes_to_i.get(node).unwrap()).collect();
    let m_list: Vec<usize> = i_list.iter().map(|i| g_i[*i]).collect();

    (i_list, m_list)
}

fn get_init_rec(egraph: &EGraph<Mdl, TensorAnalysis>, eclass: Id, added_memo: &mut HashSet<Id>, costs: &HashMap<Id, (f32, Mdl)>, nodes: &mut Vec<Mdl>) {
    let id = egraph.find(eclass);

    if !added_memo.contains(&id) {
        let (_, best_node) = match costs.get(&id) {
            Some(result) => result.clone(),
            None => panic!("Failed to extract from eclass {}", id),
        };
        best_node.for_each(|child| get_init_rec(egraph, child, added_memo, costs, nodes));
        nodes.push(best_node);
        added_memo.insert(id);
    }
}