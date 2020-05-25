use egg::{rewrite as rw, *};

define_language! {
    pub enum Model {
        "ewadd"     = Ewadd([Id; 2]),
        "ewmul"     = Ewmul([Id; 2]),
        "smul"      = Smul([Id; 2]),
        "transpose" = Transpose(Id),
        "matmul"    = Matmul([Id; 2]),
        "conv2d"    = Conv2d([Id; 6]),
        "enlarge"   = Enlarge([Id; 2]),
        "relu"      = Relu(Id),
        "poolavg"   = Poolavg([Id; 4]),
        "poolmax"   = Poolmax([Id; 4]),
        "concat"    = Concat([Id; 3]),
        "split_0"   = Split0([Id; 2]),
        "split_1"   = Split1([Id; 2]),
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
        rw!("conv-is-bilinear-1"              ; "(smul (conv2d ?sx ?sy ?p Anone ?x ?y) ?w) "                            => "(conv2d ?sx ?sy ?p Anone (smul ?x ?w) ?y)"),
        rw!("conv-is-bilinear-2"              ; "(conv2d ?sx ?sy ?p Anone ?x (ewadd ?y ?z)) "                           => "(ewadd (conv2d ?sx ?sy ?p Anone ?x ?y) (conv2d ?sx ?sy ?p Anone ?x ?z))"),
        rw!("conv-is-bilinear-3"              ; "(conv2d ?sx ?sy ?p Anone (ewadd ?x ?y) ?z) "                           => "(ewadd (conv2d ?sx ?sy ?p Anone ?x ?z) (conv2d ?sx ?sy ?p Anone ?y ?z))"),
        rw!("enlarge-convolution-kernel"      ; "(conv2d ?sx ?sy Psame ?c ?x ?y) "                                      => "(conv2d ?sx ?sy Psame ?c ?x (enlarge ?k ?y))"),
        rw!("operator-commutativity-4"        ; "(conv2d ?sx ?sy ?p Arelu ?x ?y) "                                      => "(relu (conv2d ?sx ?sy ?p Anone ?x ?y))"),
        rw!("conv-with-Arelu-applies-relu"    ; "(relu (transpose ?x)) "                                                => "(transpose (relu ?x))"),
        rw!("pooling-by-conv.-with-Cpool"     ; "(conv2d ?sx ?sy ?p Anone ?x (Cpool ?k)) "                              => "(poolavg ?k ?s ?p ?x)"),
        rw!("identity-kernel"                 ; "(conv2d 1 1 Psame Anone ?x (Iconv ?k)) "                               => "?x"),
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
        rw!("concatenation-and-conv.-2"       ; "(conv2d ?sx ?sy ?p Anone (concat 1 ?x ?z) (concat 1 ?y ?w)) "          => "(ewadd (conv2d ?sx ?sy ?p Anone ?x ?y) (conv2d ?sx ?sy ?p Anone ?z ?w))"),
        rw!("concatenation-and-pooling-0"     ; "(concat 1 (poolavg ?k ?s ?p ?x) (poolavg ?k ?s ?p ?y)) "               => "(poolavg ?k ?s ?p (concat 1 ?x ?y))"),
        rw!("concatenation-and-pooling-1"     ; "(concat 0 (poolmax ?k ?s ?p ?x) (poolmax ?k ?s ?p ?y)) "               => "(poolmax ?k ?s ?p (concat 0 ?x ?y))"),
        rw!("concatenation-and-pooling-2"     ; "(concat 1 (poolmax ?k ?s ?p ?x) (poolmax ?k ?s ?p ?y)) "               => "(poolmax ?k ?s ?p (concat 1 ?x ?y))"),
]}

extern crate pest;
#[macro_use]
extern crate pest_derive;
use pest::{Parser, iterators::Pair};

#[derive(Parser)]
#[grammar = "equation.pest"]
pub struct EqParser;

fn parse_exp(e: Pair<Rule>) -> String {
        match e.as_rule() {
                Rule::name => e.as_str().to_owned(),
                Rule::expr => parse_exp(e.into_inner().next().unwrap()),
                Rule::apply => {
                        let mut inner_rules = e.into_inner();
                        let op = parse_exp(inner_rules.next().unwrap());
                        let args = parse_exp(inner_rules.next().unwrap());
                        format!("({} {})", op, args)
                },
                Rule::args => {
                        let arg_ss: Vec<_> = e
                                .into_inner()
                                .map(parse_exp)
                                .collect();
                        arg_ss.join(" ")
                },
                _ => unreachable!()
        }
}

fn main() {
        let mut egraph = EGraph::<Model, ()>::default();
        let lhs = "(conv2d 1 1 0 0 \
         (ewadd i_12 (ewadd i_10 i_11)) \
         (ewadd i_12 (ewadd i_10 i_11)))".parse().unwrap();
        let rhs = "(conv2d 1 1 0 0 \
         (ewadd i_11 (ewadd i_10 i_12)) \
         (ewadd i_11 (ewadd i_10 i_12)))".parse().unwrap();
        egraph.add_expr(&lhs);
        egraph.add_expr(&rhs);

        let runner = Runner::default().with_egraph(egraph).run(&rules());
        assert!(!runner.egraph.equivs(&lhs, &rhs).is_empty());

        let eq = EqParser::parse(
                Rule::eq,
                "matmul(matmul(input_1,input_4),input_5)==matmul(input_1,matmul(input_4,input_5))")
                .expect("parse error")
                .next().unwrap();
        match eq.as_rule() {
                Rule::eq => {
                        let mut inner_rules = eq.into_inner();
                        let lhs = parse_exp(inner_rules.next().unwrap());
                        let rhs = parse_exp(inner_rules.next().unwrap());
                        println!("{} == {}", lhs, rhs);
                }
                _ => unreachable!(),
        }
}
