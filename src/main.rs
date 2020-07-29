use clap::{App, Arg};
use egg::*;
use std::collections::HashMap;
use std::env::*;
use std::fs::*;
use std::time::*;
use std::time::{Duration, Instant};
use tamago::benchnet;
use tamago::model::*;
use tamago::nasrnn;
use tamago::optimize::*;
use tamago::resnet50;
use tamago::resnext50;
use tamago::rewrites::*;
use tamago::testnet;
use tamago::{parse::*, verify::*};

fn main() {
    // Parse arguments
    let matches = App::new("Tamago")
        .arg(
            Arg::with_name("mode")
                .short("m")
                .long("mode")
                .takes_value(true)
                .help("Mode to run, can be verify, optimize, test, convert"),
        )
        .arg(
            Arg::with_name("model")
                .short("d")
                .long("model")
                .takes_value(true)
                .help("Specify a pre-defined model to optimize"),
        )
        .arg(
            Arg::with_name("rules")
                .short("r")
                .long("rules")
                .takes_value(true)
                .help("Provide a file with rewrite rules"),
        )
        .arg(
            Arg::with_name("out_file")
                .short("o")
                .long("out_file")
                .takes_value(true)
                .help("Provide a output file name"),
        )
        .arg(
            Arg::with_name("model_file")
                .short("f")
                .long("model_file")
                .takes_value(true)
                .help("Provide a file with the input model"),
        )
        .arg(
            Arg::with_name("multi_rules")
                .short("t")
                .long("multi_rules")
                .takes_value(true)
                .help("File with multi-pattern rules. Every two lines belong to one multi-pattern rule"),
        )
        .get_matches();

    let run_mode = matches.value_of("mode").unwrap_or("optimize");
    println!("Running mode is: {}", run_mode);

    match run_mode {
        "optimize" => optimize(matches),
        "verify" => prove_taso_rules(matches),
        "test" => test(matches),
        "convert" => convert_learned_rules(matches),
        _ => panic!("Running mode not supported"),
    }
}

fn convert_learned_rules(matches: clap::ArgMatches) {
    env_logger::init();

    let file = matches
        .value_of("rules")
        .expect("Pls supply taso rules file.");
    let outf = matches.value_of("out_file").unwrap_or("converted.txt");
    let taso_rules = read_to_string(file).expect("Something went wrong reading the file");

    let converted = parse_and_convert(&taso_rules);

    write(outf, converted).expect("Unable to write file");
}

fn test(matches: clap::ArgMatches) {
    env_logger::init();

    let start = match matches.value_of("model") {
        Some(model_name) => match model_name {
            "resnet50" => resnet50::get_resnet50(),
            "testnet" => testnet::get_testnet(),
            "benchnet" => benchnet::get_benchnet(),
            "nasrnn" => nasrnn::get_nasrnn(),
            "resnext50" => resnext50::get_resnext50(),
            _ => panic!("The model name is not supported"),
        },
        None => {
            let model_file = matches
                .value_of("model_file")
                .expect("Pls supply input graph file.");
            let input_graph =
                read_to_string(model_file).expect("Something went wrong reading the model file");
            input_graph.parse().unwrap()
        }
    };

    let runner_start = Runner::<Mdl, TensorAnalysis, ()>::default().with_expr(&start);
    runner_start
        .egraph
        .dot()
        .to_svg("target/start.svg")
        .unwrap();
    let time_start = get_full_graph_runtime(&runner_start);
    println!("Start graph runtime: {}", time_start);
}

/// Main procedure to run optimization
///
/// Gets input graph and rewrite rules; runs saturation with TensorAnalysis dealing with metadata; runs
/// greedy extraction with TensorCost getting the cost per node/op; evaluates
/// full graph runtime of the starting graph and extracted graph.
fn optimize(matches: clap::ArgMatches) {
    env_logger::init();

    // Get input graph and rules
    let rule_file = matches
        .value_of("rules")
        .expect("Pls supply rewrite rules file.");
    // learned_rules are the learned rules from TASO, pre_defined_rules are the hand-specified rules from TASO
    let learned_rules = read_to_string(rule_file).expect("Something went wrong reading the rule file");
    let pre_defined_rules = PRE_DEFINED_RULES.iter().map(|&x| x);
    let split_rules: Vec<&str> = learned_rules.split("\n").chain(pre_defined_rules).collect();
    let rules = rules_from_str(split_rules);

    let start = match matches.value_of("model") {
        Some(model_name) => match model_name {
            "resnet50" => resnet50::get_resnet50(),
            "testnet" => testnet::get_testnet(),
            "benchnet" => benchnet::get_benchnet(),
            "nasrnn" => nasrnn::get_nasrnn(),
            "resnext50" => resnext50::get_resnext50(),
            _ => panic!("The model name is not supported"),
        },
        None => {
            let model_file = matches
                .value_of("model_file")
                .expect("Pls supply input graph file.");
            let input_graph =
                read_to_string(model_file).expect("Something went wrong reading the model file");
            input_graph.parse().unwrap()
        }
    };

    // Get multi-pattern rules. learned_rules are the learned rules from TASO, pre_defined_multi are the hand-specified rules from TASO
    let multi_patterns = match matches.value_of("multi_rules") {
        Some(rule_file) => {
            let learned_rules = read_to_string(rule_file).expect("Something went wrong reading the rule file");
            let pre_defined_multi = PRE_DEFINED_MULTI.iter().map(|&x| x);
            let multi_rules: Vec<&str> = learned_rules.split("\n").chain(pre_defined_multi).collect();
            MultiPatterns::with_rules(multi_rules)
        },
        None => {
            let multi_rules: Vec<&str> = PRE_DEFINED_MULTI.iter().map(|&x| x).collect();
            MultiPatterns::with_rules(multi_rules)
        },
    };

    // Run saturation
    let time_limit_sec = Duration::new(10, 0);
    let iter_limit = 10;

    let runner = Runner::<Mdl, TensorAnalysis, ()>::default()
        .with_time_limit(time_limit_sec)
        .with_iter_limit(iter_limit)
        .with_expr(&start)
        .with_hook(move |runner| multi_patterns.run_one(runner));
    let start_time = Instant::now();
    let runner = runner.run(&rules[..]);
    let duration = start_time.elapsed();

    println!("Runner complete!");
    println!("  Nodes: {}", runner.egraph.total_size());
    println!("  Classes: {}", runner.egraph.number_of_classes());
    println!("  Stopped: {:?}", runner.stop_reason.unwrap());
    println!("  Time taken: {:?}", duration);

    // Save egraph
    let (egraph, root) = (runner.egraph, runner.roots[0]);
    egraph.dot().to_svg("target/tamago.svg").unwrap();

    // Run extraction
    let tnsr_cost = TensorCost { egraph: &egraph };
    let start_time = Instant::now();
    let mut extractor = Extractor::new(&egraph, tnsr_cost);
    let (best_cost, best) = extractor.find_best(root);
    let duration = start_time.elapsed();

    println!("Extractor complete!");
    println!("  Time taken: {:?}", duration);
    println!("  Best cost: {:?}", best_cost);

    // Evaluation starting and extracted graph runtime, save graphs
    let runner_start = Runner::<Mdl, TensorAnalysis, ()>::default().with_expr(&start);
    runner_start
        .egraph
        .dot()
        .to_svg("target/start.svg")
        .unwrap();
    let time_start = get_full_graph_runtime(&runner_start);
    println!("Start graph runtime: {}", time_start);

    let runner_ext = Runner::<Mdl, TensorAnalysis, ()>::default().with_expr(&best);
    runner_ext.egraph.dot().to_svg("target/ext.svg").unwrap();
    let time_ext = get_full_graph_runtime(&runner_ext);
    println!("Extracted graph runtime: {}", time_ext);
}

fn get_full_graph_runtime(runner: &Runner<Mdl, TensorAnalysis, ()>) -> f32 {
    let mut g = runner.egraph.analysis.graph.borrow_mut();
    unsafe {
        // This is calling TASO's preprocess_weights function before evaluating full graph
        // run time. It removes op that has only weights as its inputs. Since TASO only cares
        // about inference time, such ops can be pre-computed
        let processed_g = g.preprocess_weights();
        (*processed_g).run()
    }
}

fn prove_taso_rules(matches: clap::ArgMatches) {
    env_logger::init();

    let file = matches
        .value_of("rules")
        .expect("Pls supply taso rules file.");
    let taso_rules = read_to_string(file).expect("Something went wrong reading the file");

    println!("Parsing rules...");
    let initial = parse_rules(&taso_rules);
    println!("Parsed rules!");

    let mut to_prove = initial.clone();
    while !to_prove.is_empty() {
        let n_before = to_prove.len();
        to_prove = verify(&to_prove);
        let n_proved = n_before - to_prove.len();
        println!("Proved {} on this trip", n_proved);
        if n_proved == 0 {
            println!("\nCouldn't prove {} rule(s)", to_prove.len());
            for pair in &to_prove {
                let i = initial.iter().position(|p| p == pair).unwrap();
                println!("  {}: {} => {}", i, pair.0, pair.1);
            }
            break;
        }
    }
}
