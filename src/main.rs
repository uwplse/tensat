use egg::{rewrite as rw, *};

define_language! {
    pub enum Model {
        "ewadd"     = Ewadd([Id; 2]),
        "ewmul"     = Ewmul([Id; 2]),
        "smul"      = Smul([Id; 2]),
        "transpose" = Transpose(Id),
        "matmul"    = Matmul([Id; 2]),
        "conv"      = Conv([Id; 5]),
        "enlarge"   = Enlarge([Id; 2]),
        "relu"      = Relu(Id),
        "poolavg"   = Poolavg([Id; 4]),
        "poolmax"   = Poolmax([Id; 4]),
        "concat"    = Concat([Id; 3]),
        "split0"    = Split0([Id; 2]),
        "split1"    = Split1([Id; 2]),
        "Cpool"     = Cpool(Id),
        "Iconv"     = Iconv(Id),
        "Anone"     = Anone,
        "Arelu"     = Arelu,
        "Psame"     = Psame,
        "Imatmul"   = Imatmul,
        "Iewmul"    = Iewmul,
        Num(i32),
        Var(String),
    }
}

// TODO each rule should also have a reverse for optimization
#[rustfmt::skip]
pub fn rules() -> Vec<Rewrite<Model, ()>> { vec![
        rw!("ewadd-is-associative"            ; "(ewadd ?x (ewadd ?y ?z)) "                               => "(ewadd (ewadd ?x ?y) ?z)"),
        rw!("ewadd-is-commutative"            ; "(ewadd ?x ?y) "                                          => "(ewadd ?y ?x)"),
        rw!("ewmul-is-associative"            ; "(ewmul ?x (ewmul ?y ?z)) "                               => "(ewmul (ewmul ?x ?y) ?z)"),
        rw!("ewmul-is-commutative"            ; "(ewmul ?x ?y) "                                          => "(ewmul ?y ?x)"),
        rw!("distributivity-0"                ; "(ewmul (ewadd ?x ?y) ?z) "                               => "(ewadd (ewmul ?x ?z) (ewmul ?y ?z))"),
        rw!("smul-is-associative"             ; "(smul (smul ?x ?y) ?w) "                                 => "(smul ?x  (smul ?y ?w))"),
        rw!("distributivity-1"                ; "(smul (ewadd ?x ?y) ?w) "                                => "(ewadd (smul ?x ?w)  (smul ?y ?w))"),
        rw!("operator-commutativity-0"        ; "(smul (ewmul ?x ?y) ?w) "                                => "(ewmul ?x  (smul ?y ?w))"),
        rw!("transpose-is-its-own-inverse"    ; "(transpose (transpose ?x)) "                             => "?x"),
        rw!("operator-commutativity-1"        ; "(transpose (ewadd ?x ?y)) "                              => "(ewadd (transpose ?x)  (transpose ?y))"),
        rw!("operator-commutativity-2"        ; "(transpose (ewmul ?x ?y)) "                              => "(ewmul (transpose ?x)  (transpose ?y))"),
        rw!("operator-commutativity-3"        ; "(smul (transpose ?x) ?w) "                               => "(transpose (smul ?x ?w))"),
        rw!("matmul-is-associative"           ; "(matmul ?x (matmul ?y ?z)) "                             => "(matmul (matmul ?x ?y) ?z)"),
        rw!("matmul-is-linear-0"              ; "(smul (matmul ?x ?y) ?w) "                               => "(matmul ?x  (smul ?y ?w))"),
        rw!("matmul-is-linear-1"              ; "(matmul ?x (ewadd ?y ?z)) "                              => "(ewadd (matmul ?x ?y) (matmul ?x ?z))"),
        rw!("matmul-and-transpose"            ; "(transpose (matmul ?x ?y)) "                             => "(matmul (transpose ?y)  (transpose ?x))"),
        rw!("conv-is-bilinear-0"              ; "(conv ?s ?p ?c (smul ?x ?w) ?y) "                        => "(conv ?s ?p ?c ?x (smul ?y ?w))"),
        rw!("conv-is-bilinear-1"              ; "(smul (conv ?s ?p Anone ?x ?y) ?w) "                     => "(conv ?s ?p Anone (smul ?x ?w) ?y)"),
        rw!("conv-is-bilinear-2"              ; "(conv ?s ?p Anone ?x (ewadd ?y ?z)) "                    => "(ewadd (conv ?s ?p Anone ?x ?y) (conv ?s ?p Anone ?x ?z))"),
        rw!("conv-is-bilinear-3"              ; "(conv ?s ?p Anone (ewadd ?x ?y) ?z) "                    => "(ewadd (conv ?s ?p Anone ?x ?z) (conv ?s ?p Anone ?y ?z))"),
        rw!("enlarge-convolution-kernel"      ; "(conv ?s Psame ?c ?x ?y) "                               => "(conv ?s Psame ?c ?x (enlarge ?k ?y))"),
        rw!("operator-commutativity-4"        ; "(conv ?s ?p Arelu ?x ?y) "                               => "(relu (conv ?s ?p Anone ?x ?y))"),
        rw!("conv-with-Arelu-applies-relu"    ; "(relu (transpose ?x)) "                                  => "(transpose (relu ?x))"),
        rw!("pooling-by-conv.-with-Cpool"     ; "(conv ?s ?p Anone ?x (Cpool ?k)) "                       => "(poolavg ?k ?s ?p ?x)"),
        rw!("identity-kernel"                 ; "(conv 1 Psame Anone ?x (Iconv ?k)) "                     => "?x"),
        rw!("identity-matrix"                 ; "(matmul ?x   Imatmul ) "                                 => "?x"),
        rw!("ewmul-identity"                  ; "(ewmul ?x Iewmul) "                                      => "?x"),
        rw!("split-definition-0"              ; "(split0 ?a (concat ?a ?x ?y)) "                          => "?x"),
        rw!("split-definition-1"              ; "(split1 ?a (concat ?a ?x ?y)) "                          => "?y"),
        rw!("geometry-of-concatenation"       ; "(concat 0 (concat 1 ?x ?y) (concat 1 ?z ?w)) "           => "(concat 1 (concat 0 ?x ?z) (concat 0 ?y ?w))"),
        rw!("operator-commutativity-5"        ; "(concat ?a (smul ?x ?w) (smul ?y ?w)) "                  => "(smul (concat ?a ?x ?y) ?w)"),
        rw!("operator-commutativity-6"        ; "(concat ?a (ewadd ?x ?y) (ewadd ?z ?w)) "                => "(ewadd (concat ?a ?x ?z) (concat ?a ?y ?w))"),
        rw!("operator-commutativity-7"        ; "(concat ?a (ewmul ?x ?y) (ewmul ?z ?w)) "                => "(ewmul (concat ?a ?x ?z) (concat ?a ?y ?w))"),
        rw!("operator-commutativity-8"        ; "(concat ?a (relu ?x) (relu ?y)) "                        => "(relu (concat ?a ?x ?y))"),
        rw!("concatenation-and-transpose"     ; "(concat 1 (transpose ?x) (transpose ?y)) "               => "(transpose (concat 0 ?x ?y))"),
        rw!("concatenation-and-matrix-mul.-0" ; "(concat 1 (matmul ?x ?y) (matmul ?x ?z)) "               => "(matmul ?x (concat 1 ?y ?z))"),
        rw!("concatenation-and-matrix-mul.-1" ; "(matmul (concat 1 ?x ?z) (concat 0 ?y ?w)) "             => "(ewadd (matmul ?x ?y) (matmul ?z ?w))"),
        rw!("concatenation-and-conv.-0"       ; "(concat 0 (conv ?s ?p ?c ?x ?z) (conv ?s ?p ?c ?y ?z)) " => "(conv ?s ?p ?c (concat 0 ?x ?y) ?z)"),
        rw!("concatenation-and-conv.-1"       ; "(concat 1 (conv ?s ?p ?c ?x ?y) (conv ?s ?p ?c ?x ?z)) " => "(conv ?s ?p ?c ?x (concat 0 ?y ?z))"),
        rw!("concatenation-and-conv.-2"       ; "(conv ?s ?p Anone (concat 1 ?x ?z) (concat 1 ?y ?w)) "   => "(ewadd (conv ?s ?p Anone ?x ?y) (conv ?s ?p Anone ?z ?w))"),
        rw!("concatenation-and-pooling-0"     ; "(concat 1 (poolavg ?k ?s ?p ?x) (poolavg ?k ?s ?p ?y)) " => "(poolavg ?k ?s ?p (concat 1 ?x ?y))"),
        rw!("concatenation-and-pooling-1"     ; "(concat 0 (poolmax ?k ?s ?p ?x) (poolmax ?k ?s ?p ?y)) " => "(poolmax ?k ?s ?p (concat 0 ?x ?y))"),
        rw!("concatenation-and-pooling-2"     ; "(concat 1 (poolmax ?k ?s ?p ?x) (poolmax ?k ?s ?p ?y)) " => "(poolmax ?k ?s ?p (concat 1 ?x ?y))"),
]}

fn main() {
        let end = "(matmul (matmul input_1 input_4) input_5)".parse().unwrap();
        let start = "(matmul input_1 (matmul input_4 input_5))".parse().unwrap();
        let runner = Runner::default().with_expr(&start).run(&rules());
        let (egraph, root) = (runner.egraph, runner.roots[0]);

        let mut extractor = Extractor::new(&egraph, AstSize);
        let (best_cost, best) = extractor.find_best(root);

        println!("Smallest expression is {} with size {}.", best.pretty(80), best_cost);
        assert!(!egraph.equivs(&start, &end).is_empty());
}
