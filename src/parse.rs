use crate::input::*;
use crate::model::*;
use egg::*;
use pest::{iterators::Pair, Parser};
use root::taso::*;
use std::collections::HashMap;

#[derive(pest_derive::Parser)]
#[grammar = "equation.pest"]
pub struct EqParser;

pub fn parse_exp(e: Pair<Rule>) -> String {
    match e.as_rule() {
        Rule::name => e.as_str().to_owned(),
        Rule::expr => parse_exp(e.into_inner().next().unwrap()),
        Rule::apply => {
            let mut inner_rules = e.into_inner();
            let op = parse_exp(inner_rules.next().unwrap());
            let args = parse_exp(inner_rules.next().unwrap());
            format!("({} {})", op, args)
        }
        Rule::args => {
            let arg_ss: Vec<_> = e.into_inner().map(parse_exp).collect();
            arg_ss.join(" ")
        }
        _ => unreachable!(),
    }
}

pub fn parse_eq(e: Pair<Rule>) -> (RecExpr<Mdl>, RecExpr<Mdl>) {
    match e.as_rule() {
        Rule::eq => {
            let mut inner_rules = e.into_inner();
            let lhs = parse_exp(inner_rules.next().unwrap());
            let rhs = parse_exp(inner_rules.next().unwrap());
            (lhs.parse().unwrap(), rhs.parse().unwrap())
        }
        _ => unreachable!(),
    }
}

pub fn parse_rules(rs_s: &str) -> Vec<(RecExpr<Mdl>, RecExpr<Mdl>)> {
    let rs = EqParser::parse(Rule::prog, rs_s)
        .expect("parse error")
        .next()
        .unwrap();
    match rs.as_rule() {
        Rule::prog => rs.into_inner().map(parse_eq).collect(),
        _ => unreachable!(),
    }
}

pub fn convert_eq(e: Pair<Rule>) -> String {
    match e.as_rule() {
        Rule::eq => {
            let mut inner_rules = e.into_inner();
            let lhs = parse_exp(inner_rules.next().unwrap());
            let rhs = parse_exp(inner_rules.next().unwrap());
            let eq_str = format!("{}=>{}", lhs, rhs);
            str::replace(&eq_str, "input", "?input")
        }
        _ => unreachable!(),
    }
}

pub fn parse_and_convert(rs_s: &str) -> String {
    let rs = EqParser::parse(Rule::prog, rs_s)
        .expect("parse error")
        .next()
        .unwrap();
    match rs.as_rule() {
        Rule::prog => {
            let converted_rules: Vec<String> = rs.into_inner().map(convert_eq).collect();
            let joined = converted_rules.join("\n");
            joined
        }
        _ => unreachable!(),
    }
}

// parses a serialized model from taso
// see tests/parse.rs for an example
pub fn parse_model(rs_s: &str) -> GraphConverter {
    let mut ls = rs_s.lines();
    let mut g = GraphConverter::default();
    let mut nodes: HashMap<usize, Vec<TensorInfo>> = HashMap::new();
    loop {
        if let Some(l) = ls.next() {
            // node id
            let guid = l.parse::<usize>().unwrap();
            // the operator
            let op = ls.next().unwrap().parse::<u32>().unwrap();
            // children; each child has an id and an index;
            // the index is almost always 0, except when the child
            // is a split it may be 0 or 1 (indicating left or right)
            let deps: Vec<Vec<usize>> = ls
                .next()
                .unwrap()
                .split(",")
                .map(|c_s| c_s.split(":").map(|c| c.parse().unwrap()).collect())
                .collect();
            // parameters
            let params: Vec<i32> = ls
                .next()
                .unwrap()
                .split(",")
                .map(|p_s| p_s.parse().unwrap())
                .collect();
            // node is really a vec, because split may return two outputs
            let node: Vec<TensorInfo> = match op {
                OpType_OP_INPUT => vec![g.new_input(&params)],
                OpType_OP_WEIGHT => vec![g.new_weight(&params)],
                OpType_OP_MATMUL => vec![g.matmul(
                    nodes[&deps[0][0]][deps[0][1]],
                    nodes[&deps[1][0]][deps[1][1]],
                )],
                OpType_OP_RELU => vec![g.relu(nodes[&deps[0][0]][deps[0][1]])],
                OpType_OP_RESHAPE => vec![g.reshape(nodes[&deps[0][0]][deps[0][1]], &params)],
                OpType_OP_TRANSPOSE => {
                    vec![g.transpose(nodes[&deps[0][0]][deps[0][1]], &params[..3], params[3] != 0)]
                }
                OpType_OP_SPLIT => todo!(), // reference 'Split' case in taso/examples/load_model.py
                o => panic!("{} not yet implemented", o),
            };
            nodes.insert(guid, node);
        } else {
            break;
        }
    }
    g
}
