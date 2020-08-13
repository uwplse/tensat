use crate::model::*;
use egg::{rewrite as rw, *};
use itertools::Itertools;
use root::taso::*;
use std::collections::{HashMap, HashSet};
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
        let eqn: Vec<&str> = rule.split("=>").collect();
        let lhs: Pattern<Mdl> = eqn[0].parse().unwrap();
        let rhs: Pattern<Mdl> = eqn[1].parse().unwrap();
        let rule_name = format!("rule{}", pos);
        rule_vec.push(rw!(rule_name; lhs => { CheckApply {pat: rhs} }));
    }
    rule_vec
}

/// Hand specified normal rules from TASO
#[rustfmt::skip]
pub static PRE_DEFINED_RULES: &[&str] = &[
    "(conv2d 1 1 0 0 ?input_1 ?input_2)=>(conv2d 1 1 0 0 ?input_1 (merge ?input_2 2))",
    "(conv2d 1 1 0 2 ?input_1 ?input_2)=>(conv2d 1 1 0 2 ?input_1 (merge ?input_2 2))",
    "(conv2d 2 2 0 0 ?input_1 ?input_2)=>(conv2d 2 2 0 0 ?input_1 (merge ?input_2 2))",
    "(conv2d 2 2 0 2 ?input_1 ?input_2)=>(conv2d 2 2 0 2 ?input_1 (merge ?input_2 2))",
];

/// Hand specified multi-pattern rules from TASO
#[rustfmt::skip]
pub static PRE_DEFINED_MULTI: &[&str] = &[
    "(conv2d 1 1 0 0 ?input_1 ?input_2)=>(split_0 (split 1 (conv2d 1 1 0 0 ?input_1 (concat 0 4 (enlarge ?input_2 ?input_3) ?input_3))))",
    "(conv2d 1 1 0 0 ?input_1 ?input_3)=>(split_1 (split 1 (conv2d 1 1 0 0 ?input_1 (concat 0 4 (enlarge ?input_2 ?input_3) ?input_3))))",
    "(conv2d 1 1 0 2 ?input_1 ?input_2)=>(split_0 (split 1 (conv2d 1 1 0 2 ?input_1 (concat 0 4 (enlarge ?input_2 ?input_3) ?input_3))))",
    "(conv2d 1 1 0 2 ?input_1 ?input_3)=>(split_1 (split 1 (conv2d 1 1 0 2 ?input_1 (concat 0 4 (enlarge ?input_2 ?input_3) ?input_3))))",
];

/// Struct for passing results in the recursive function check_pat
///
/// Similar as ValTnsr for TensorAnalysis, but with tnsr being the object
/// rather than pointer, to make memory working correctly with recursive
/// function.
struct TData {
    pub dtype: DataKind,
    pub val: i32,
    pub tnsr: Option<Tensor>,
    pub tnsr_2: Option<Tensor>,
}

impl Default for TData {
    fn default() -> Self {
        TData {
            tnsr: None,
            tnsr_2: None,
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
            let t_data = if egraph[cid].data.dtype == DataKind::Tnsr {
                TData {
                    dtype: egraph[cid].data.dtype,
                    val: egraph[cid].data.val,
                    tnsr: unsafe { Some((*egraph[cid].data.meta).clone()) },
                    tnsr_2: None,
                }
            } else {
                // A variable cannot refer to a TnsrTuple, so we don't need that case
                TData {
                    dtype: egraph[cid].data.dtype,
                    val: egraph[cid].data.val,
                    tnsr: None,
                    tnsr_2: None,
                }
            };
            return (true, Some(cid), t_data);
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
                            let t_data = match egraph[id].data.dtype {
                                DataKind::Tnsr => TData {
                                    dtype: egraph[id].data.dtype,
                                    val: egraph[id].data.val,
                                    tnsr: unsafe { Some((*egraph[id].data.meta).clone()) },
                                    tnsr_2: None,
                                },
                                DataKind::TnsrTuple => TData {
                                    dtype: egraph[id].data.dtype,
                                    val: egraph[id].data.val,
                                    tnsr: unsafe { Some((*egraph[id].data.meta).clone()) },
                                    tnsr_2: unsafe { Some((*egraph[id].data.meta_2).clone()) },
                                },
                                _ => TData {
                                    dtype: egraph[id].data.dtype,
                                    val: egraph[id].data.val,
                                    tnsr: None,
                                    tnsr_2: None,
                                },
                            };
                            return (true, looked, t_data);
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
                            tnsr_2: None,
                        };
                        (true, None, t_data)
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
                                    tnsr_2: None,
                                };
                                (true, None, t_data)
                            }
                        }
                    }

                    Mdl::Tanh(_a) => {
                        let a_t_data = &results[0].2;
                        assert!(a_t_data.dtype == DataKind::Tnsr);
                        let t_a = a_t_data.tnsr.unwrap();

                        unsafe {
                            let op = (*g.model).get_or_create_activation(t_a, OpType_OP_TANH, true);
                            if op == Op_INVALID_OP {
                                let default_data: TData = Default::default();
                                (false, None, default_data)
                            } else {
                                let t = (*op.ptr).outputs[0].clone();
                                let t_data = TData {
                                    dtype: DataKind::Tnsr,
                                    val: 0,
                                    tnsr: Some(t),
                                    tnsr_2: None,
                                };
                                (true, None, t_data)
                            }
                        }
                    }

                    Mdl::Sigmoid(_a) => {
                        let a_t_data = &results[0].2;
                        assert!(a_t_data.dtype == DataKind::Tnsr);
                        let t_a = a_t_data.tnsr.unwrap();

                        unsafe {
                            let op =
                                (*g.model).get_or_create_activation(t_a, OpType_OP_SIGMOID, true);
                            if op == Op_INVALID_OP {
                                let default_data: TData = Default::default();
                                (false, None, default_data)
                            } else {
                                let t = (*op.ptr).outputs[0].clone();
                                let t_data = TData {
                                    dtype: DataKind::Tnsr,
                                    val: 0,
                                    tnsr: Some(t),
                                    tnsr_2: None,
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
                            let op = (*g.model).get_or_create_conv2d(
                                t_inpt, t_wght, stride_h, stride_w, padding, activation,
                            );
                            if op == Op_INVALID_OP {
                                let default_data: TData = Default::default();
                                (false, None, default_data)
                            } else {
                                let t = (*op.ptr).outputs[0].clone();
                                let t_data = TData {
                                    dtype: DataKind::Tnsr,
                                    val: 0,
                                    tnsr: Some(t),
                                    tnsr_2: None,
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
                            let op = (*g.model).get_or_create_element(OpType_OP_EW_ADD, &t_a, &t_b);
                            if op == Op_INVALID_OP {
                                let default_data: TData = Default::default();
                                (false, None, default_data)
                            } else {
                                let t = (*op.ptr).outputs[0].clone();
                                let t_data = TData {
                                    dtype: DataKind::Tnsr,
                                    val: 0,
                                    tnsr: Some(t),
                                    tnsr_2: None,
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
                            let op = (*g.model).get_or_create_element(OpType_OP_EW_MUL, &t_a, &t_b);
                            if op == Op_INVALID_OP {
                                let default_data: TData = Default::default();
                                (false, None, default_data)
                            } else {
                                let t = (*op.ptr).outputs[0].clone();
                                let t_data = TData {
                                    dtype: DataKind::Tnsr,
                                    val: 0,
                                    tnsr: Some(t),
                                    tnsr_2: None,
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
                            let op = (*g.model).get_or_create_matmul(t_a, t_b, activation);
                            if op == Op_INVALID_OP {
                                let default_data: TData = Default::default();
                                (false, None, default_data)
                            } else {
                                let t = (*op.ptr).outputs[0].clone();
                                let t_data = TData {
                                    dtype: DataKind::Tnsr,
                                    val: 0,
                                    tnsr: Some(t),
                                    tnsr_2: None,
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

                            let mut need_copy = [false, false];
                            unsafe {
                                let op = (*g.model).get_or_create_concat(
                                    axis,
                                    2,
                                    ptr,
                                    need_copy.as_mut_ptr(),
                                );

                                if op == Op_INVALID_OP {
                                    let default_data: TData = Default::default();
                                    (false, None, default_data)
                                } else {
                                    let t = (*op.ptr).outputs[0].clone();
                                    let t_data = TData {
                                        dtype: DataKind::Tnsr,
                                        val: 0,
                                        tnsr: Some(t),
                                        tnsr_2: None,
                                    };
                                    (true, None, t_data)
                                }
                            }
                        }
                    }

                    Mdl::Merge([_weight, _count]) => {
                        // Check types
                        let _weight_data = &results[0].2;
                        let _count_data = &results[1].2;
                        assert!(_count_data.dtype == DataKind::Scalar);
                        assert!(_weight_data.dtype == DataKind::Tnsr);

                        // Get arguments
                        let t_weight = _weight_data.tnsr.unwrap();
                        let count = _count_data.val;

                        // Try creating op
                        unsafe {
                            let op = (*g.model).get_or_create_merge_gconv(&t_weight, count);
                            if op == Op_INVALID_OP {
                                let default_data: TData = Default::default();
                                (false, None, default_data)
                            } else {
                                let t = (*op.ptr).outputs[0].clone();
                                let t_data = TData {
                                    dtype: DataKind::Tnsr,
                                    val: 0,
                                    tnsr: Some(t),
                                    tnsr_2: None,
                                };
                                (true, None, t_data)
                            }
                        }
                    }

                    Mdl::Split([_axis, _inpt]) => {
                        // Check types
                        let _axis_data = &results[0].2;
                        let _inpt_data = &results[1].2;
                        assert!(_axis_data.dtype == DataKind::Scalar);
                        assert!(_inpt_data.dtype == DataKind::Tnsr);

                        // Get arguments
                        let t_inpt = _inpt_data.tnsr.unwrap();
                        let axis = _axis_data.val;

                        // Try creating op
                        unsafe {
                            let op = (*g.model).get_or_create_split1(&t_inpt, axis, 2);
                            if op == Op_INVALID_OP {
                                let default_data: TData = Default::default();
                                (false, None, default_data)
                            } else {
                                let t_1 = (*op.ptr).outputs[0].clone();
                                let t_2 = (*op.ptr).outputs[1].clone();
                                let t_data = TData {
                                    dtype: DataKind::TnsrTuple,
                                    val: 0,
                                    tnsr: Some(t_1),
                                    tnsr_2: Some(t_2),
                                };
                                (true, None, t_data)
                            }
                        }
                    }

                    Mdl::Split0(_inpt) => {
                        // Check types
                        let _inpt_data = &results[0].2;
                        assert!(_inpt_data.dtype == DataKind::TnsrTuple);

                        let t_data = TData {
                            dtype: DataKind::Tnsr,
                            val: 0,
                            tnsr: _inpt_data.tnsr,
                            tnsr_2: None,
                        };

                        (true, None, t_data)
                    }

                    Mdl::Split1(_inpt) => {
                        // Check types
                        let _inpt_data = &results[0].2;
                        assert!(_inpt_data.dtype == DataKind::TnsrTuple);

                        let t_data = TData {
                            dtype: DataKind::Tnsr,
                            val: 0,
                            tnsr: _inpt_data.tnsr_2,
                            tnsr_2: None,
                        };

                        (true, None, t_data)
                    }

                    Mdl::Enlarge([_a, _b]) => {
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
                            let op = (*g.model).get_or_create_enlarge(t_a, t_b);
                            if op == Op_INVALID_OP {
                                let default_data: TData = Default::default();
                                (false, None, default_data)
                            } else {
                                let t = (*op.ptr).outputs[0].clone();
                                let t_data = TData {
                                    dtype: DataKind::Tnsr,
                                    val: 0,
                                    tnsr: Some(t),
                                    tnsr_2: None,
                                };
                                (true, None, t_data)
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

/// Struct for storing information on how each pattern maps to its canonical version
#[derive(Debug)]
struct MapToCanonical {
    /// Index into MultiPatterns.canonical_src_pat. Points to the canonical version.
    index: usize,
    /// Mapping from variable in this pattern to variable in the canonical pattern.
    var_map: HashMap<egg::Var, egg::Var>,
}

/// Struct for the multi-pattern rules. In charge of searching for matches and
/// applying the rewrite.
#[derive(Debug)]
pub struct MultiPatterns {
    /// Vec of (src_1, src_2, dst_1, dst_2, symmetric)
    rules: Vec<(Pattern<Mdl>, Pattern<Mdl>, Pattern<Mdl>, Pattern<Mdl>, bool)>,
    /// Vec of all unique canonical source patterns (for src_1's and src_2's)
    canonical_src_pat: Vec<Pattern<Mdl>>,
    /// Mapping information for each src pattern. The order is the same as in rules
    src_pat_maps: Vec<(MapToCanonical, MapToCanonical)>,
    /// Whether to allow cycles in EGraph
    no_cycle: bool,
    /// Number of iterations to run multi-pattern rules
    iter_limit: usize,
}

impl MultiPatterns {
    /// Construct a MultiPatterns with rules. Each multi-pattern rule contains two matched outputs.
    ///
    /// # Parameters
    ///
    /// - `rules`: every adjacent pair of entries should belong to the same multi-pattern rule.
    pub fn with_rules(rules: Vec<(&str, bool)>, no_cycle: bool, iter_limit: usize) -> MultiPatterns {
        assert!(rules.len() % 2 == 0);

        let mut multi_rules =
            Vec::<(Pattern<Mdl>, Pattern<Mdl>, Pattern<Mdl>, Pattern<Mdl>, bool)>::new();
        let mut canonical_pats = Vec::<Pattern<Mdl>>::new();
        let mut src_pat_maps = Vec::<(MapToCanonical, MapToCanonical)>::new();

        let mut canonicalize_and_add = |pat: &Pattern<Mdl>| {
            let (pat_canonical, pat_var_map) = canonicalize(pat);

            let index_found = canonical_pats.iter().position(|x| *x == pat_canonical);
            let pat_index = index_found
                .or_else(|| {
                    canonical_pats.push(pat_canonical);
                    Some(canonical_pats.len() - 1)
                })
                .unwrap();
            MapToCanonical {
                index: pat_index,
                var_map: pat_var_map,
            }
        };

        let get_pats = |rule: &str| {
            rule.split("=>")
                .map(|x| x.parse().unwrap())
                .next_tuple()
                .unwrap()
        };

        for i in 0..(rules.len() / 2) {
            let (src_1, dst_1) = get_pats(rules[2 * i].0);
            let (src_2, dst_2) = get_pats(rules[2 * i + 1].0);

            let src_1_map = canonicalize_and_add(&src_1);
            let src_2_map = canonicalize_and_add(&src_2);

            assert!(rules[2 * i].1 == rules[2 * i + 1].1);
            let symmetric = rules[2 * i].1;

            multi_rules.push((src_1, src_2, dst_1, dst_2, symmetric));
            src_pat_maps.push((src_1_map, src_2_map));
        }

        println!("Number of canonicalized {:?}", canonical_pats.len());

        MultiPatterns {
            rules: multi_rules,
            canonical_src_pat: canonical_pats,
            src_pat_maps: src_pat_maps,
            no_cycle: no_cycle,
            iter_limit: iter_limit,
        }
    }

    /// Search and apply all multi-pattern rules for one iteration
    ///
    /// This function is used as hook function to egg::Runner. It first searches for matches
    /// of all canonicalized source patterns. Then for all compatible substitutions found,
    /// it checks and applies the dst patterns. It won't apply if src_1 and src_2 matches with
    /// the same eclass. It always returns Ok()
    pub fn run_one(&self, runner: &mut Runner<Mdl, TensorAnalysis, ()>) -> Result<(), String> {
        if runner.iterations.len() < self.iter_limit {
            println!("Run one");
            // Construct Vec to store matches for each canonicalized pattern
            let matches: Vec<Vec<SearchMatches>> = self
                .canonical_src_pat
                .iter()
                .map(|x| x.search(&runner.egraph))
                .collect();

            // For each multi rule
            for (i, rule) in self.rules.iter().enumerate() {
                let map_1 = &self.src_pat_maps[i].0;
                let map_2 = &self.src_pat_maps[i].1;
                // If the rule is fully symmetrical
                if map_1.index == map_2.index && rule.4 {
                    let matches_both = &matches[map_1.index];
                    for (i, match_1) in matches_both.iter().enumerate() {
                        for match_2 in (&matches_both[(i + 1)..]).iter() {
                            if match_1.eclass == match_2.eclass {
                                // We don't want to apply multi-pattern rules on the same eclass
                                continue;
                            }
                            self.apply_match_pair(rule, match_1, match_2, map_1, map_2, runner);
                        }
                    }
                } else {
                    let matches_1 = &matches[map_1.index];
                    let matches_2 = &matches[map_2.index];
                    for match_1 in matches_1 {
                        for match_2 in matches_2 {
                            if match_1.eclass == match_2.eclass {
                                // We don't want to apply multi-pattern rules on the same eclass
                                continue;
                            }
                            self.apply_match_pair(rule, match_1, match_2, map_1, map_2, runner);
                        }
                    }
                }
            }

            runner.egraph.rebuild();
        }

        Ok(())
    }

    /// Apply a rule with a pair of matches for its src patterns
    fn apply_match_pair(
        &self,
        rule: &(Pattern<Mdl>, Pattern<Mdl>, Pattern<Mdl>, Pattern<Mdl>, bool),
        match_1: &SearchMatches,
        match_2: &SearchMatches,
        map_1: &MapToCanonical,
        map_2: &MapToCanonical,
        runner: &mut Runner<Mdl, TensorAnalysis, ()>,
    ) {
        let mut descendents: HashMap<Id, HashSet<Id>> = Default::default();
        for subst_1 in &match_1.substs {
            for subst_2 in &match_2.substs {
                // De-canonicalize the substitutions
                let subst_1_dec = decanonicalize(subst_1, &map_1.var_map);
                let subst_2_dec = decanonicalize(subst_2, &map_2.var_map);
                // Check if two substitutions have matching shared variables
                if compatible(&subst_1_dec, &subst_2_dec, &map_1.var_map) {
                    // If so, merge two substitutions
                    let merged_subst = merge_subst(subst_1_dec, subst_2_dec, &map_1.var_map);
                    // check_pat on both dst patterns
                    if check_pat(rule.2.ast.as_ref(), &mut runner.egraph, &merged_subst).0 {
                        if check_pat(rule.3.ast.as_ref(), &mut runner.egraph, &merged_subst).0 {
                            let cycle_check_passed = if self.no_cycle {
                                check_cycle(
                                    &runner.egraph,
                                    &merged_subst,
                                    &map_1.var_map,
                                    &map_2.var_map,
                                    match_1.eclass,
                                    match_2.eclass,
                                    &mut descendents,
                                )
                            } else {
                                true
                            };
                            if cycle_check_passed {
                                // apply dst patterns, union
                                let id_1 = rule.2.apply_one(
                                    &mut runner.egraph,
                                    match_1.eclass,
                                    &merged_subst,
                                )[0];
                                runner.egraph.union(id_1, match_1.eclass);
                                let id_2 = rule.3.apply_one(
                                    &mut runner.egraph,
                                    match_2.eclass,
                                    &merged_subst,
                                )[0];
                                runner.egraph.union(id_2, match_2.eclass);
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Returns true if there will not be a cycle
fn check_cycle(
    egraph: &EGraph<Mdl, TensorAnalysis>,
    input_subst: &Subst,
    var_map_1: &HashMap<egg::Var, egg::Var>,
    var_map_2: &HashMap<egg::Var, egg::Var>,
    out_class_1: Id,
    out_class_2: Id,
    descendents: &mut HashMap<Id, HashSet<Id>>,
) -> bool {
    // Get all input eclass IDs
    let input_ids: HashSet<Id> = var_map_1.iter().chain(var_map_2.iter()).map(|(var, _)| *input_subst.get(*var).unwrap()).collect();
    // Get a map from eclass IDs to eclass
    let id_to_class: HashMap<Id, &EClass<Mdl, ValTnsr>> = egraph.classes().map(|class| (class.id, class)).collect();
    // Check descendents of the input eclasses
    for id in input_ids.iter() {
        get_descendents(egraph, *id, &id_to_class, descendents);
        let descendents_input = descendents.get(id).unwrap();
        if descendents_input.contains(&out_class_1) || descendents_input.contains(&out_class_2) {
            return false;
        }
    }
    true
}

fn get_descendents(egraph: &EGraph<Mdl, TensorAnalysis>, eclass: Id, id_to_class: &HashMap<Id, &EClass<Mdl, ValTnsr>>, descendents: &mut HashMap<Id, HashSet<Id>>) {
    match descendents.get(&eclass) {
        Some(desc) => (),
        None => {
            let class = id_to_class.get(&eclass).unwrap();
            let mut result_desc = HashSet::<Id>::new();
            for node in class.iter() {
                for child in node.children().iter() {
                    get_descendents(egraph, *child, id_to_class, descendents);
                    let child_desc = descendents.get(child).unwrap();
                    result_desc = result_desc.union(child_desc).map(|&id| id).collect();
                    result_desc.insert(*child);
                }
            }
            descendents.insert(eclass, result_desc);
        },
    }
}

/// Canonicalize a pattern
///
/// This function constructs a canonicalized pattern for a given pattern. It sequentially
/// replaces each unique variable with ?i_0, ?i_1 ...
///
/// # Parameters
///
/// - `pat`: the pattern to canonicalize. See egg::Pattern for more info
///
/// # Returns
///
/// - Pattern<Mdl>: the canonicalized pattern
/// - HashMap<egg::Var, egg::Var>: Mapping from variable in the original pattern to
///     variable in the canonical pattern.
fn canonicalize(pat: &Pattern<Mdl>) -> (Pattern<Mdl>, HashMap<egg::Var, egg::Var>) {
    let mut var_map = HashMap::<egg::Var, egg::Var>::new();
    let mut count = 0;
    let substituted: Vec<_> = pat
        .ast
        .as_ref()
        .iter()
        .cloned()
        .map(|x| match x {
            ENodeOrVar::ENode(_) => x,
            ENodeOrVar::Var(v) => {
                let var = var_map.entry(v).or_insert_with(|| {
                    let name = format!("?i_{}", count);
                    count += 1;
                    name.parse().unwrap()
                });
                ENodeOrVar::Var(*var)
            }
        })
        .collect();

    let ast = RecExpr::from(substituted);
    let canonical_pat = Pattern::<Mdl>::from(ast);
    (canonical_pat, var_map)
}

/// Merge two substitutions
///
/// This function merges two substitutions. The merged one contains substitutions (egg::Var -> Id)
/// from both input substitutions.
///
/// # Parameters
///
/// - `subst_1`: substitution to be merged
/// - `subst_2`: substitution to be merged
/// - `var_map_1`: the keys of this map should be all Var in subst_1. It is just for providing the Vars
fn merge_subst(
    subst_1: Subst,
    mut subst_2: Subst,
    var_map_1: &HashMap<egg::Var, egg::Var>,
) -> Subst {
    for (var, _) in var_map_1.iter() {
        let id_1 = subst_1.get(*var).unwrap();
        subst_2.insert(*var, *id_1);
    }
    subst_2
}

/// Decanonicalize a substitution
///
/// Create a decanonicalized substitution by replacing the variables in the canonical substitution
/// with the original variables
///
/// # Parameters
///
/// - `subst`: The substitution using the canonicalized variables
/// - `var_map`: Mapping from variable in the original pattern to variable in the canonical pattern.
fn decanonicalize(subst: &Subst, var_map: &HashMap<egg::Var, egg::Var>) -> Subst {
    let mut new_subst: Subst = Default::default();
    for (orig_var, canonical_var) in var_map.iter() {
        new_subst.insert(*orig_var, *subst.get(*canonical_var).unwrap());
    }
    new_subst
}

/// Check if the shared variables between two substitutions point to the same eclass Id.
///
/// # Parameters
///
/// - `subst_1`: substitution to be checked
/// - `subst_2`: substitution to be checked
/// - `var_map_1`: the keys of this map should be all Var in subst_1. It is just for providing the Vars
///
/// # Returns
///
///   Return true if all corresponding shared variables between two substitutions point to the
///   same eclass Id's.
fn compatible(subst_1: &Subst, subst_2: &Subst, var_map_1: &HashMap<egg::Var, egg::Var>) -> bool {
    for (var, _) in var_map_1.iter() {
        let id_1 = subst_1.get(*var).unwrap();
        if let Some(id_2) = subst_2.get(*var) {
            if id_1 != id_2 {
                return false;
            }
        }
    }
    return true;
}
