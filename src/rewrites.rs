use crate::model::*;
use egg::{rewrite as rw, *};
use root::taso::*;
use std::convert::TryInto;
use std::time::{Duration, Instant};

// TODO egg now provides bidirectional rules whic should cut down
// this list in half.
#[rustfmt::skip]
pub fn rules<A: Analysis<Mdl>>() -> Vec<Rewrite<Mdl, A>> { vec![
        rw!("ewadd-is-associative"            ; "(ewadd ?x (ewadd ?y ?z)) "                                             => "(ewadd (ewadd ?x ?y) ?z)"),
        rw!("ewadd-is-commutative"            ; "(ewadd ?x ?y) "                                                        => "(ewadd ?y ?x)"),
        rw!("ewmul-is-associative"            ; "(ewmul ?x (ewmul ?y ?z)) "                                             => "(ewmul (ewmul ?x ?y) ?z)"),
        rw!("ewmul-is-commutative"            ; "(ewmul ?x ?y) "                                                        => "(ewmul ?y ?x)"),
        rw!("distributivity-0"                ; "(ewmul (ewadd ?x ?y) ?z) "                                             => "(ewadd (ewmul ?x ?z) (ewmul ?y ?z))"),
        rw!("smul-is-associative"             ; "(smul (smul ?x ?y) ?w) "                                               => "(smul ?x  (smul ?y ?w))"),
        rw!("distributivity-1"                ; "(smul (ewadd ?x ?y) ?w) "                                              => "(ewadd (smul ?x ?w)  (smul ?y ?w))"),
        rw!("operator-commutativity-0"        ; "(smul (ewmul ?x ?y) ?w) "                                              => "(ewmul ?x  (smul ?y ?w))"),
        rw!("transpose-is-its-own-inverse"    ; "(transpose (transpose ?x)) "                                           => "?x"),
        rw!("operator-commutativity-1"        ; "(transpose (ewadd ?x ?y)) "                                            => "(ewadd (transpose ?x)  (transpose ?y))"),
        rw!("operator-commutativity-2"        ; "(transpose (ewmul ?x ?y)) "                                            => "(ewmul (transpose ?x)  (transpose ?y))"),
        rw!("operator-commutativity-3"        ; "(smul (transpose ?x) ?w) "                                             => "(transpose (smul ?x ?w))"),
        rw!("matmul-is-associative"           ; "(matmul ?x (matmul ?y ?z)) "                                           => "(matmul (matmul ?x ?y) ?z)"),
        rw!("matmul-is-linear-0"              ; "(smul (matmul ?x ?y) ?w) "                                             => "(matmul ?x  (smul ?y ?w))"),
        rw!("matmul-is-linear-1"              ; "(matmul ?x (ewadd ?y ?z)) "                                            => "(ewadd (matmul ?x ?y) (matmul ?x ?z))"),
        rw!("matmul-and-transpose"            ; "(transpose (matmul ?x ?y)) "                                           => "(matmul (transpose ?y)  (transpose ?x))"),
        rw!("conv-is-bilinear-0"              ; "(conv2d ?sx ?sy ?p ?c (smul ?x ?w) ?y) "                               => "(conv2d ?sx ?sy ?p ?c ?x (smul ?y ?w))"),
        rw!("conv-is-bilinear-1"              ; "(smul (conv2d ?sx ?sy ?p 0 ?x ?y) ?w) "                            => "(conv2d ?sx ?sy ?p 0 (smul ?x ?w) ?y)"),
        rw!("conv-is-bilinear-2"              ; "(conv2d ?sx ?sy ?p 0 ?x (ewadd ?y ?z)) "                           => "(ewadd (conv2d ?sx ?sy ?p 0 ?x ?y) (conv2d ?sx ?sy ?p 0 ?x ?z))"),
        rw!("conv-is-bilinear-3"              ; "(conv2d ?sx ?sy ?p 0 (ewadd ?x ?y) ?z) "                           => "(ewadd (conv2d ?sx ?sy ?p 0 ?x ?z) (conv2d ?sx ?sy ?p 0 ?y ?z))"),
        //rw!("enlarge-convolution-kernel"      ; "(conv2d ?sx ?sy 0 ?c ?x ?y) "                                      => "(conv2d ?sx ?sy 0 ?c ?x (enlarge ?kx ?ky ?y))"),
        rw!("operator-commutativity-4"        ; "(conv2d ?sx ?sy ?p 2 ?x ?y) "                                      => "(relu (conv2d ?sx ?sy ?p 0 ?x ?y))"),
        rw!("conv-with-2-applies-relu"    ; "(relu (transpose ?x)) "                                                => "(transpose (relu ?x))"),
        // rw!("pooling-by-conv.-with-Cpool"     ; "(conv2d ?sx ?sy ?p 0 ?x (Cpool ?kx ?ky)) "                              => "(poolavg ?kx ?ky ?sx ?sy ?p ?x)"),
        rw!("const_iconv-and-const_pool"      ; "(poolavg ?kx ?ky 1 1 0 (Iconv ?kx ?ky)) "                              => "(Cpool ?kx ?ky)"),
        rw!("identity-kernel"                 ; "(conv2d 1 1 0 0 ?x (Iconv ?kx ?ky)) "                               => "?x"),
        rw!("identity-matrix"                 ; "(matmul ?x   Imatmul ) "                                               => "?x"),
        rw!("ewmul-identity"                  ; "(ewmul ?x Iewmul) "                                                    => "?x"),
        rw!("split-definition-0"              ; "(split_0 ?a (concat ?a ?x ?y)) "                                       => "?x"),
        rw!("split-definition-1"              ; "(split_1 ?a (concat ?a ?x ?y)) "                                       => "?y"),
        rw!("geometry-of-concatenation"       ; "(concat 0 (concat 1 ?x ?y) (concat 1 ?z ?w)) "                         => "(concat 1 (concat 0 ?x ?z) (concat 0 ?y ?w))"),
        rw!("operator-commutativity-5"        ; "(concat ?a (smul ?x ?w) (smul ?y ?w)) "                                => "(smul (concat ?a ?x ?y) ?w)"),
        rw!("operator-commutativity-6"        ; "(concat ?a (ewadd ?x ?y) (ewadd ?z ?w)) "                              => "(ewadd (concat ?a ?x ?z) (concat ?a ?y ?w))"),
        rw!("operator-commutativity-7"        ; "(concat ?a (ewmul ?x ?y) (ewmul ?z ?w)) "                              => "(ewmul (concat ?a ?x ?z) (concat ?a ?y ?w))"),
        rw!("operator-commutativity-8"        ; "(concat ?a (relu ?x) (relu ?y)) "                                      => "(relu (concat ?a ?x ?y))"),
        rw!("concatenation-and-transpose"     ; "(concat 1 (transpose ?x) (transpose ?y)) "                             => "(transpose (concat 0 ?x ?y))"),
        rw!("concatenation-and-matrix-mul.-0" ; "(concat 1 (matmul ?x ?y) (matmul ?x ?z)) "                             => "(matmul ?x (concat 1 ?y ?z))"),
        rw!("concatenation-and-matrix-mul.-1" ; "(matmul (concat 1 ?x ?z) (concat 0 ?y ?w)) "                           => "(ewadd (matmul ?x ?y) (matmul ?z ?w))"),
        rw!("concatenation-and-conv.-0"       ; "(concat 0 (conv2d ?sx ?sy ?p ?c ?x ?z) (conv2d ?sx ?sy ?p ?c ?y ?z)) " => "(conv2d ?sx ?sy ?p ?c (concat 0 ?x ?y) ?z)"),
        rw!("concatenation-and-conv.-1"       ; "(concat 1 (conv2d ?sx ?sy ?p ?c ?x ?y) (conv2d ?sx ?sy ?p ?c ?x ?z)) " => "(conv2d ?sx ?sy ?p ?c ?x (concat 0 ?y ?z))"),
        rw!("concatenation-and-conv.-2"       ; "(conv2d ?sx ?sy ?p 0 (concat 1 ?x ?z) (concat 1 ?y ?w)) "          => "(ewadd (conv2d ?sx ?sy ?p 0 ?x ?y) (conv2d ?sx ?sy ?p 0 ?z ?w))"),
        rw!("concatenation-and-pooling-0"     ; "(concat 1 (poolavg ?kx ?ky ?sx ?sy ?p ?x) (poolavg ?kx ?ky ?sx ?sy ?p ?y)) "               => "(poolavg ?kx ?ky ?sx ?sy ?p (concat 1 ?x ?y))"),
        rw!("concatenation-and-pooling-1"     ; "(concat 0 (poolmax ?kx ?ky ?sx ?sy ?p ?x) (poolmax ?kx ?ky ?sx ?sy ?p ?y)) "               => "(poolmax ?kx ?ky ?sx ?sy ?p (concat 0 ?x ?y))"),
        rw!("concatenation-and-pooling-2"     ; "(concat 1 (poolmax ?kx ?ky ?sx ?sy ?p ?x) (poolmax ?kx ?ky ?sx ?sy ?p ?y)) "               => "(poolmax ?kx ?ky ?sx ?sy ?p (concat 1 ?x ?y))"),
        // inverse
        rw!("-ewadd-is-associative"            ;"(ewadd (ewadd ?x ?y) ?z)"                                                => "(ewadd ?x (ewadd ?y ?z)) "                                             ),
        rw!("-ewadd-is-commutative"            ;"(ewadd ?y ?x)"                                                           => "(ewadd ?x ?y) "                                                        ),
        rw!("-ewmul-is-associative"            ;"(ewmul (ewmul ?x ?y) ?z)"                                                => "(ewmul ?x (ewmul ?y ?z)) "                                             ),
        rw!("-ewmul-is-commutative"            ;"(ewmul ?y ?x)"                                                           => "(ewmul ?x ?y) "                                                        ),
        rw!("-distributivity-0"                ;"(ewadd (ewmul ?x ?z) (ewmul ?y ?z))"                                     => "(ewmul (ewadd ?x ?y) ?z) "                                             ),
        rw!("-smul-is-associative"             ;"(smul ?x  (smul ?y ?w))"                                                 => "(smul (smul ?x ?y) ?w) "                                               ),
        rw!("-distributivity-1"                ;"(ewadd (smul ?x ?w)  (smul ?y ?w))"                                      => "(smul (ewadd ?x ?y) ?w) "                                              ),
        rw!("-operator-commutativity-0"        ;"(ewmul ?x  (smul ?y ?w))"                                                => "(smul (ewmul ?x ?y) ?w) "                                              ),
        rw!("-transpose-is-its-own-inverse"    ;"?x"                                                                      => "(transpose (transpose ?x)) "                                           ),
        rw!("-operator-commutativity-1"        ;"(ewadd (transpose ?x)  (transpose ?y))"                                  => "(transpose (ewadd ?x ?y)) "                                            ),
        rw!("-operator-commutativity-2"        ;"(ewmul (transpose ?x)  (transpose ?y))"                                  => "(transpose (ewmul ?x ?y)) "                                            ),
        rw!("-operator-commutativity-3"        ;"(transpose (smul ?x ?w))"                                                => "(smul (transpose ?x) ?w) "                                             ),
        rw!("-matmul-is-associative"           ;"(matmul (matmul ?x ?y) ?z)"                                              => "(matmul ?x (matmul ?y ?z)) "                                           ),
        rw!("-matmul-is-linear-0"              ;"(matmul ?x  (smul ?y ?w))"                                               => "(smul (matmul ?x ?y) ?w) "                                             ),
        rw!("-matmul-is-linear-1"              ;"(ewadd (matmul ?x ?y) (matmul ?x ?z))"                                   => "(matmul ?x (ewadd ?y ?z)) "                                            ),
        rw!("-matmul-and-transpose"            ;"(matmul (transpose ?y)  (transpose ?x))"                                 => "(transpose (matmul ?x ?y)) "                                           ),
        rw!("-conv-is-bilinear-0"              ;"(conv2d ?sx ?sy ?p ?c ?x (smul ?y ?w))"                                  => "(conv2d ?sx ?sy ?p ?c (smul ?x ?w) ?y) "                               ),
        rw!("-conv-is-bilinear-1"              ;"(conv2d ?sx ?sy ?p 0 (smul ?x ?w) ?y)"                               => "(smul (conv2d ?sx ?sy ?p 0 ?x ?y) ?w) "                            ),
        rw!("-conv-is-bilinear-2"              ;"(ewadd (conv2d ?sx ?sy ?p 0 ?x ?y) (conv2d ?sx ?sy ?p 0 ?x ?z))" => "(conv2d ?sx ?sy ?p 0 ?x (ewadd ?y ?z)) "                           ),
        rw!("-conv-is-bilinear-3"              ;"(ewadd (conv2d ?sx ?sy ?p 0 ?x ?z) (conv2d ?sx ?sy ?p 0 ?y ?z))" => "(conv2d ?sx ?sy ?p 0 (ewadd ?x ?y) ?z) "                           ),
        rw!("-enlarge-convolution-kernel"      ;"(conv2d ?sx ?sy 0 ?c ?x (enlarge ?kx ?ky ?y))"                            => "(conv2d ?sx ?sy 0 ?c ?x ?y) "                                      ),
        rw!("-operator-commutativity-4"        ;"(relu (conv2d ?sx ?sy ?p 0 ?x ?y))"                                  => "(conv2d ?sx ?sy ?p 2 ?x ?y) "                                      ),
        rw!("-conv-with-2-applies-relu"    ;"(transpose (relu ?x))"                                                   => "(relu (transpose ?x)) "                                                ),
        rw!("-pooling-by-conv.-with-Cpool"     ;"(poolavg ?kx ?ky ?sx ?sy ?p ?x)"                                                   => "(conv2d ?sx ?sy ?p 0 ?x (Cpool ?kx ?ky)) "                              ),
        rw!("-const_iconv-and-const_pool"      ;"(Cpool ?kx ?ky)"                              => "(poolavg ?kx ?ky 1 1 0 (Iconv ?kx ?ky))"),
        // rw!("-identity-kernel"                 ;"?x"                                                                      => "(conv2d 1 1 0 0 ?x (Iconv ?k)) "                               ),
        rw!("-identity-matrix"                 ;"?x"                                                                      => "(matmul ?x   Imatmul ) "                                               ),
        rw!("-ewmul-identity"                  ;"?x"                                                                      => "(ewmul ?x Iewmul) "                                                    ),
        // rw!("-split-definition-00"              ;"?x"                                                                      => "(split_0 1 (concat 1 ?x ?y)) "                                       ),
        // rw!("-split-definition-01"              ;"?x"                                                                      => "(split_0 0 (concat 0 ?x ?y)) "                                       ),
        // rw!("-split-definition-10"              ;"?y"                                                                      => "(split_1 0 (concat 0 ?x ?y)) "                                       ),
        // rw!("-split-definition-11"              ;"?y"                                                                      => "(split_1 1 (concat 1 ?x ?y)) "                                       ),
        rw!("-geometry-of-concatenation"       ;"(concat 1 (concat 0 ?x ?z) (concat 0 ?y ?w))"                            => "(concat 0 (concat 1 ?x ?y) (concat 1 ?z ?w)) "                         ),
        rw!("-operator-commutativity-5"        ;"(smul (concat ?a ?x ?y) ?w)"                                             => "(concat ?a (smul ?x ?w) (smul ?y ?w)) "                                ),
        rw!("-operator-commutativity-6"        ;"(ewadd (concat ?a ?x ?z) (concat ?a ?y ?w))"                             => "(concat ?a (ewadd ?x ?y) (ewadd ?z ?w)) "                              ),
        rw!("-operator-commutativity-7"        ;"(ewmul (concat ?a ?x ?z) (concat ?a ?y ?w))"                             => "(concat ?a (ewmul ?x ?y) (ewmul ?z ?w)) "                              ),
        rw!("-operator-commutativity-8"        ;"(relu (concat ?a ?x ?y))"                                                => "(concat ?a (relu ?x) (relu ?y)) "                                      ),
        rw!("-concatenation-and-transpose"     ;"(transpose (concat 0 ?x ?y))"                                            => "(concat 1 (transpose ?x) (transpose ?y)) "                             ),
        rw!("-concatenation-and-matrix-mul.-0" ;"(matmul ?x (concat 1 ?y ?z))"                                            => "(concat 1 (matmul ?x ?y) (matmul ?x ?z)) "                             ),
        rw!("-concatenation-and-matrix-mul.-1" ;"(ewadd (matmul ?x ?y) (matmul ?z ?w))"                                   => "(matmul (concat 1 ?x ?z) (concat 0 ?y ?w)) "                           ),
        rw!("-concatenation-and-conv.-0"       ;"(conv2d ?sx ?sy ?p ?c (concat 0 ?x ?y) ?z)"                              => "(concat 0 (conv2d ?sx ?sy ?p ?c ?x ?z) (conv2d ?sx ?sy ?p ?c ?y ?z)) " ),
        rw!("-concatenation-and-conv.-1"       ;"(conv2d ?sx ?sy ?p ?c ?x (concat 0 ?y ?z))"                              => "(concat 1 (conv2d ?sx ?sy ?p ?c ?x ?y) (conv2d ?sx ?sy ?p ?c ?x ?z)) " ),
        rw!("-concatenation-and-conv.-2"       ;"(ewadd (conv2d ?sx ?sy ?p 0 ?x ?y) (conv2d ?sx ?sy ?p 0 ?z ?w))" => "(conv2d ?sx ?sy ?p 0 (concat 1 ?x ?z) (concat 1 ?y ?w)) "          ),
        rw!("-concatenation-and-pooling-0"     ;"(poolavg ?kx ?ky ?sx ?sy ?p (concat 1 ?x ?y))"                                     => "(concat 1 (poolavg ?kx ?ky ?sx ?sy ?p ?x) (poolavg ?kx ?ky ?sx ?sy ?p ?y)) "               ),
        rw!("-concatenation-and-pooling-1"     ;"(poolmax ?kx ?ky ?sx ?sy ?p (concat 0 ?x ?y))"                                     => "(concat 0 (poolmax ?kx ?ky ?sx ?sy ?p ?x) (poolmax ?kx ?ky ?sx ?sy ?p ?y)) "               ),
        rw!("-concatenation-and-pooling-2"     ;"(poolmax ?kx ?ky ?sx ?sy ?p (concat 1 ?x ?y))"                                     => "(concat 1 (poolmax ?kx ?ky ?sx ?sy ?p ?x) (poolmax ?kx ?ky ?sx ?sy ?p ?y)) "               ),
]}

pub fn rules_from_str(rs: Vec<&str>) -> Vec<Rewrite<Mdl, TensorAnalysis>> {
    let mut rule_vec = Vec::new();
    for (pos, rule) in rs.iter().enumerate() {
        //println!("{:?}", rule);
        let eqn: Vec<&str> = rule.split("=>").collect();
        let lhs: Pattern<Mdl> = eqn[0].parse().unwrap();
        let rhs: Pattern<Mdl> = eqn[1].parse().unwrap();
        let rule_name = format!("rule{}", pos);
        rule_vec.push(rw!(rule_name; lhs => { CheckApply {pat: rhs} }));
    }
    rule_vec
}

/// Struct for passing results in the recursive function check_pat
///
/// Similar as ValTnsr for TensorAnalysis, but with tnsr being the object
/// rather than pointer, to make memory working correctly with recursive
/// function.
struct TData {
    pub dtype: DataKind,
    pub val: i32,
    pub tnsr: Option<Tensor>,
}

impl Default for TData {
    fn default() -> Self {
        TData {
            tnsr: None,
            val: Default::default(),
            dtype: Default::default(),
        }
    }
}

impl PartialEq for Op {
    fn eq(&self, other: &Self) -> bool {
        self.guid == other.guid && self.ptr == other.ptr
    }
}

/// Custom struct implementing the Applier trait, checking the new nodes to
/// construct are all valid before actually apply.
#[derive(Debug, Clone, PartialEq)]
struct CheckApply {
    /// the pattern of the right hand side of the rewrite rule, the one
    /// to be constructed.
    pat: Pattern<Mdl>,
}

impl Applier<Mdl, TensorAnalysis> for CheckApply {
    /// Apply the pattern once. Check the new nodes are valid before actually
    /// apply. See Applier trait in egg for more information.
    fn apply_one(
        &self,
        egraph: &mut EGraph<Mdl, TensorAnalysis>,
        matched_id: Id,
        subst: &Subst,
    ) -> Vec<Id> {
        if check_pat(self.pat.ast.as_ref(), egraph, subst).0 {
            self.pat.apply_one(egraph, matched_id, subst)
        } else {
            vec![]
        }
    }

    fn vars(&self) -> Vec<Var> {
        self.pat.vars()
    }
}

/// Check if all the new nodes to create in the pattern is valid.
///
/// This function does the checking recursively.
///
/// # Parameters
///
/// - `pat`: the AST representation of the pattern. See egg::Pattern for more info
/// - `egraph`: E-graph of interest
/// - `subst`: mapping variable to eclass ID. See egg::Subst for more info.
///
/// # Returns
///
/// A tuple of (bool, Option<Id>, TData) where
///
/// - bool: true if the nodes in this pattern are all valid
/// - Option<Id>: if the root node of this pattern (pat.last()) is in egraph,
///     then it is the Id of that eclass. Otherwise None
/// - TData: The TData for the root node of this pattern. This is read from
///     egraph if the root node is in egraph, otherwise constructed by calling
///     TASO functions.
fn check_pat(
    pat: &[ENodeOrVar<Mdl>],
    egraph: &mut EGraph<Mdl, TensorAnalysis>,
    subst: &Subst,
) -> (bool, Option<Id>, TData) {
    match pat.last().unwrap() {
        ENodeOrVar::Var(w) => {
            // The root node is a variable, then use subst to get metadata from egraph
            let cid = subst[*w];
            unsafe {
                let t_data = if egraph[cid].data.dtype == DataKind::Tnsr {
                    TData {
                        dtype: egraph[cid].data.dtype,
                        val: egraph[cid].data.val,
                        tnsr: Some((*egraph[cid].data.meta).clone()),
                    }
                } else {
                    TData {
                        dtype: egraph[cid].data.dtype,
                        val: egraph[cid].data.val,
                        tnsr: None,
                    }
                };
                return (true, Some(cid), t_data);
            }
        }
        ENodeOrVar::ENode(e) => {
            // The root is an enode. Recursively get checking results from its children
            let children = e.children();
            let results: Vec<(bool, Option<Id>, TData)> = children
                .iter()
                .map(|child| check_pat(&pat[..usize::from(*child) + 1], egraph, subst))
                .collect();

            // Check if any children contains invalid nodes
            let mut violated = false;
            for res in &results {
                if !res.0 {
                    violated = true;
                }
            }
            if violated {
                let default_data: TData = Default::default();
                return (false, None, default_data);
            } else {
                // Check if all children are in egraph
                let mut all_in = true;
                for res in &results {
                    let is_in = match res.1 {
                        Some(_) => true,
                        None => false,
                    };
                    if !is_in {
                        all_in = false;
                    }
                }
                if all_in {
                    // Construct enode, check if in egraph
                    let mut new_e = e.clone();
                    let new_e_ch = new_e.children_mut();
                    for (i, res) in results.iter().enumerate() {
                        new_e_ch[i] = res.1.unwrap();
                    }
                    let looked = egraph.lookup(new_e);
                    match looked {
                        Some(id) => {
                            // Get metadata from egraph
                            unsafe {
                                let t_data = if egraph[id].data.dtype == DataKind::Tnsr {
                                    TData {
                                        dtype: egraph[id].data.dtype,
                                        val: egraph[id].data.val,
                                        tnsr: Some((*egraph[id].data.meta).clone()),
                                    }
                                } else {
                                    TData {
                                        dtype: egraph[id].data.dtype,
                                        val: egraph[id].data.val,
                                        tnsr: None,
                                    }
                                };
                                return (true, looked, t_data);
                            }
                        }
                        None => (),
                    };
                }
                // root node not in egraph, compute metadata
                let mut g = egraph.analysis.graph.borrow_mut();
                let result = match e {
                    Mdl::Num(_n) => {
                        let t_data = TData {
                            dtype: DataKind::Scalar,
                            val: *_n,
                            tnsr: None,
                        };
                        (true, None, t_data)
                    }

                    Mdl::Var(_s) => {
                        let t_data = TData {
                            dtype: DataKind::Name,
                            val: 0,
                            tnsr: None,
                        };
                        (true, None, t_data)
                    }

                    Mdl::Input([_name, _dim1, _dim2, _dim3, _dim4]) => {
                        let mut dims = vec![
                            results[1].2.val,
                            results[2].2.val,
                            results[3].2.val,
                            results[4].2.val,
                        ];
                        dims.shrink_to_fit();
                        assert!(dims.len() == dims.capacity());
                        let ptr = dims.as_mut_ptr();
                        unsafe {
                            std::mem::forget(dims);
                            let inp = g.new_input(4, ptr);
                            let t_data = TData {
                                dtype: DataKind::Tnsr,
                                val: 0,
                                tnsr: Some(*inp),
                            };
                            (true, None, t_data)
                        }
                    }

                    Mdl::Relu(_a) => {
                        let a_t_data = &results[0].2;
                        assert!(a_t_data.dtype == DataKind::Tnsr);
                        let t_a = a_t_data.tnsr.unwrap();

                        unsafe {
                            let op = (*g.model).get_or_create_activation(t_a, OpType_OP_RELU, true);
                            if op == Op_INVALID_OP {
                                let default_data: TData = Default::default();
                                (false, None, default_data)
                            } else {
                                let t = (*op.ptr).outputs[0].clone();
                                let t_data = TData {
                                    dtype: DataKind::Tnsr,
                                    val: 0,
                                    tnsr: Some(t),
                                };
                                (true, None, t_data)
                            }
                        }
                    }

                    Mdl::Conv2d([_stride_h, _stride_w, _pad, _act, _inpt, _wght]) => {
                        // Check types
                        let _stride_h_data = &results[0].2;
                        let _stride_w_data = &results[1].2;
                        let _pad_data = &results[2].2;
                        let _act_data = &results[3].2;
                        let _inpt_data = &results[4].2;
                        let _wght_data = &results[5].2;
                        assert!(_stride_h_data.dtype == DataKind::Scalar);
                        assert!(_stride_w_data.dtype == DataKind::Scalar);
                        assert!(_pad_data.dtype == DataKind::Scalar);
                        assert!(_act_data.dtype == DataKind::Scalar);
                        assert!(_inpt_data.dtype == DataKind::Tnsr);
                        assert!(_wght_data.dtype == DataKind::Tnsr);

                        // Get arguments
                        let t_inpt = _inpt_data.tnsr.unwrap();
                        let t_wght = _wght_data.tnsr.unwrap();
                        let stride_h = _stride_h_data.val;
                        let stride_w = _stride_w_data.val;
                        let padding: PaddingMode = _pad_data.val.try_into().unwrap();
                        let activation: ActiMode = _act_data.val.try_into().unwrap();

                        // Try creating op
                        unsafe {
                            //let start_time = Instant::now();
                            let op = (*g.model).get_or_create_conv2d(
                                t_inpt, t_wght, stride_h, stride_w, padding, activation,
                            );
                            //let duration = start_time.elapsed();
                            //println!("  Time taken getc conv: {:?}", duration);
                            if op == Op_INVALID_OP {
                                let default_data: TData = Default::default();
                                (false, None, default_data)
                            } else {
                                let t = (*op.ptr).outputs[0].clone();
                                let t_data = TData {
                                    dtype: DataKind::Tnsr,
                                    val: 0,
                                    tnsr: Some(t),
                                };
                                (true, None, t_data)
                            }
                        }
                    }

                    Mdl::Ewadd([_a, _b]) => {
                        // Check types
                        let _a_data = &results[0].2;
                        let _b_data = &results[1].2;
                        assert!(_a_data.dtype == DataKind::Tnsr);
                        assert!(_b_data.dtype == DataKind::Tnsr);

                        // Get arguments
                        let t_a = _a_data.tnsr.unwrap();
                        let t_b = _b_data.tnsr.unwrap();

                        // Try creating op
                        unsafe {
                            //let start_time = Instant::now();
                            let op = (*g.model).get_or_create_element(OpType_OP_EW_ADD, &t_a, &t_b);
                            //let duration = start_time.elapsed();
                            //println!("  Time taken getc ele: {:?}", duration);
                            if op == Op_INVALID_OP {
                                let default_data: TData = Default::default();
                                (false, None, default_data)
                            } else {
                                let t = (*op.ptr).outputs[0].clone();
                                let t_data = TData {
                                    dtype: DataKind::Tnsr,
                                    val: 0,
                                    tnsr: Some(t),
                                };
                                (true, None, t_data)
                            }
                        }
                    }

                    Mdl::Ewmul([_a, _b]) => {
                        // Check types
                        let _a_data = &results[0].2;
                        let _b_data = &results[1].2;
                        assert!(_a_data.dtype == DataKind::Tnsr);
                        assert!(_b_data.dtype == DataKind::Tnsr);

                        // Get arguments
                        let t_a = _a_data.tnsr.unwrap();
                        let t_b = _b_data.tnsr.unwrap();

                        // Try creating op
                        unsafe {
                            //let start_time = Instant::now();
                            let op = (*g.model).get_or_create_element(OpType_OP_EW_MUL, &t_a, &t_b);
                            //let duration = start_time.elapsed();
                            //println!("  Time taken getc ele: {:?}", duration);
                            if op == Op_INVALID_OP {
                                let default_data: TData = Default::default();
                                (false, None, default_data)
                            } else {
                                let t = (*op.ptr).outputs[0].clone();
                                let t_data = TData {
                                    dtype: DataKind::Tnsr,
                                    val: 0,
                                    tnsr: Some(t),
                                };
                                (true, None, t_data)
                            }
                        }
                    }

                    Mdl::Matmul([_act, _a, _b]) => {
                        // Check types
                        let _act_data = &results[0].2;
                        let _a_data = &results[1].2;
                        let _b_data = &results[2].2;
                        assert!(_act_data.dtype == DataKind::Scalar);
                        assert!(_a_data.dtype == DataKind::Tnsr);
                        assert!(_b_data.dtype == DataKind::Tnsr);

                        // Get arguments
                        let t_a = _a_data.tnsr.unwrap();
                        let t_b = _b_data.tnsr.unwrap();
                        let activation: ActiMode = _act_data.val.try_into().unwrap();

                        // Try creating op
                        unsafe {
                            //let start_time = Instant::now();
                            let op = (*g.model).get_or_create_matmul(t_a, t_b, activation);
                            //let duration = start_time.elapsed();
                            //println!("  Time taken getc ele: {:?}", duration);
                            if op == Op_INVALID_OP {
                                let default_data: TData = Default::default();
                                (false, None, default_data)
                            } else {
                                let t = (*op.ptr).outputs[0].clone();
                                let t_data = TData {
                                    dtype: DataKind::Tnsr,
                                    val: 0,
                                    tnsr: Some(t),
                                };
                                (true, None, t_data)
                            }
                        }
                    }

                    Mdl::Concat([_axis, _ndim, _a, _b]) => {
                        // Check types
                        let _axis_data = &results[0].2;
                        let _ndim_data = &results[1].2;
                        let _a_data = &results[2].2;
                        let _b_data = &results[3].2;
                        assert!(_axis_data.dtype == DataKind::Scalar);
                        assert!(_ndim_data.dtype == DataKind::Scalar);
                        assert!(_a_data.dtype == DataKind::Tnsr);
                        assert!(_b_data.dtype == DataKind::Tnsr);

                        // Get arguments
                        let t_a = _a_data.tnsr.unwrap();
                        let t_b = _b_data.tnsr.unwrap();
                        let axis = _axis_data.val;
                        let ndim = _ndim_data.val;

                        // Try creating op
                        unsafe {
                            // Check tensor ndim
                            if t_a.numDim != ndim || t_b.numDim != ndim {
                                let default_data: TData = Default::default();
                                (false, None, default_data)
                            } else {
                                // Pass ownership to C++
                                let mut inputs = vec![t_a, t_b];
                                inputs.shrink_to_fit();
                                assert!(inputs.len() == inputs.capacity());
                                let ptr = inputs.as_mut_ptr();
                                std::mem::forget(inputs);

                                let need_copy = vec![false, false].as_mut_ptr();
                                let op = (*g.model).get_or_create_concat(axis, 2, ptr, need_copy);

                                if op == Op_INVALID_OP {
                                    let default_data: TData = Default::default();
                                    (false, None, default_data)
                                } else {
                                    let t = (*op.ptr).outputs[0].clone();
                                    let t_data = TData {
                                        dtype: DataKind::Tnsr,
                                        val: 0,
                                        tnsr: Some(t),
                                    };
                                    (true, None, t_data)
                                }
                            }
                        }
                    }

                    other => {
                        println!("{:?}", other);
                        todo!()
                    }
                };
                return result;
            }
        }
    };
}
