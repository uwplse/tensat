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
use tamago::bert;
use tamago::{parse::*, verify::*};

use std::io::Error;
use std::process::{Command, Stdio};
use std::thread;

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
        .arg(
            Arg::with_name("save_graph")
                .short("s")
                .long("save_graph")
                .takes_value(true)
                .help("Whether to save graphs as dot files. Can be: all, io, none"),
        )
        .arg(
            Arg::with_name("use_multi")
                .short("u")
                .long("use_multi")
                .help("Set this flag will enable use of multi-pattern rules"),
        )
        .arg(
            Arg::with_name("n_iter")
                .long("n_iter")
                .takes_value(true)
                .help("Max number of iterations for egg to run"),
        )
        .arg(
            Arg::with_name("n_sec")
                .long("n_sec")
                .takes_value(true)
                .help("Max number of seconds for egg to run"),
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
    create_dir_all("./tmp");
    let outf = "./tmp/test.txt";
    write(outf, "2").expect("Unable to write file");
    let child = Command::new("python")
        .args(&["extractor/test.py"])
        .spawn()
        .expect("failed to execute child");

    let output = child
        .wait_with_output()
        .expect("failed to get output");

    if output.status.success() {
        let new_num = read_to_string("./tmp/new.txt").expect("Something went wrong reading the file");
        println!("New number: {}", new_num);
    } else {
        println!("Failed");
}
}

/// Main procedure to run optimization
///
/// Gets input graph and rewrite rules; runs saturation with TensorAnalysis dealing with metadata; runs
/// greedy extraction with TensorCost getting the cost per node/op; evaluates
/// full graph runtime of the starting graph and extracted graph.
fn optimize(matches: clap::ArgMatches) {
    env_logger::init();

    // Read settings from args
    let rule_file = matches
        .value_of("rules")
        .expect("Pls supply rewrite rules file.");
    let save_graph = matches.value_of("save_graph").unwrap_or("all");
    let use_multi = matches.is_present("use_multi");

    // Get input graph and rules
    // learned_rules are the learned rules from TASO, pre_defined_rules are the hand-specified rules from TASO
    let learned_rules =
        read_to_string(rule_file).expect("Something went wrong reading the rule file");
    let pre_defined_rules = PRE_DEFINED_RULES.iter().map(|&x| x);
    let split_rules: Vec<&str> = learned_rules.split("\n").chain(pre_defined_rules).collect();
    let rules = rules_from_str(split_rules);

    let start = match matches.value_of("model") {
        Some("resnet50") => resnet50::get_resnet50(),
        Some("testnet") => testnet::get_testnet(),
        Some("benchnet") => benchnet::get_benchnet(),
        Some("nasrnn") => nasrnn::get_nasrnn(),
        Some("resnext50") => resnext50::get_resnext50(),
        Some("bert") => bert::get_bert(),
        Some(_) => panic!("The model name is not supported"),
        None => {
            let model_file = matches
                .value_of("model_file")
                .expect("Pls supply input graph file.");
            let input_graph =
                read_to_string(model_file).expect("Something went wrong reading the model file");
            input_graph.parse().unwrap()
        }
    };

    // Get multi-pattern rules. learned_rules are the learned rules from TASO,
    // pre_defined_multi are the hand-specified rules from TASO
    let multi_patterns = if let Some(rule_file) = matches.value_of("multi_rules") {
        let learned_rules =
            read_to_string(rule_file).expect("Something went wrong reading the rule file");
        let pre_defined_multi = PRE_DEFINED_MULTI.iter().map(|&x| x);
        let multi_rules: Vec<&str> = learned_rules.split("\n").chain(pre_defined_multi).collect();
        MultiPatterns::with_rules(multi_rules)
    } else {
        let multi_rules: Vec<&str> = PRE_DEFINED_MULTI.iter().map(|&x| x).collect();
        MultiPatterns::with_rules(multi_rules)
    };

    // Run saturation
    let n_sec = matches
        .value_of("n_sec")
        .map_or(10, |s| s.parse::<u64>().unwrap());
    let time_limit_sec = Duration::new(n_sec, 0);
    let iter_limit = matches
        .value_of("n_iter")
        .map_or(3, |s| s.parse::<usize>().unwrap());

    let runner = if use_multi {
        // This hook function (which applies the multi-pattern rules) will be called at the
        // beginning of each iteration in equality saturation
        Runner::<Mdl, TensorAnalysis, ()>::default()
            .with_node_limit(100000)
            .with_time_limit(time_limit_sec)
            .with_iter_limit(iter_limit)
            .with_expr(&start)
            .with_hook(move |runner| multi_patterns.run_one(runner))
    } else {
        Runner::<Mdl, TensorAnalysis, ()>::default()
            .with_node_limit(100000)
            .with_time_limit(time_limit_sec)
            .with_iter_limit(iter_limit)
            .with_expr(&start)
    };
    let start_time = Instant::now();
    let runner = runner.run(&rules[..]);
    let duration = start_time.elapsed();

    println!("Runner complete!");
    println!("  Nodes: {}", runner.egraph.total_size());
    println!("  Classes: {}", runner.egraph.number_of_classes());
    println!("  Stopped: {:?}", runner.stop_reason.unwrap());
    println!("  Time taken: {:?}", duration);
    println!("  Number of iterations: {:?}", runner.iterations.len()-1);

    let (num_enodes, num_classes, avg_nodes_per_class, num_edges) = get_stats(&runner.egraph);
    println!("  Average nodes per class: {}", avg_nodes_per_class);
    println!("  Number of edges: {}", num_edges);

    // Save egraph
    let (egraph, root) = (runner.egraph, runner.roots[0]);
    if save_graph == "all" {
        egraph.dot().to_svg("target/tamago.svg").unwrap();
    }

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
    let runner_ext = Runner::<Mdl, TensorAnalysis, ()>::default().with_expr(&best);

    if save_graph != "none" {
        runner_start
            .egraph
            .dot()
            .to_svg("target/start.svg")
            .unwrap();
        runner_ext.egraph.dot().to_svg("target/ext.svg").unwrap();
    }

    let time_start = get_full_graph_runtime(&runner_start);
    println!("Start graph runtime: {}", time_start);

    let time_ext = get_full_graph_runtime(&runner_ext);
    println!("Extracted graph runtime: {}", time_ext);
}

/// This function gets the following stats:
///     Total number of enodes
///     Total number of eclasses
///     Average number of enodes per class
///     Total number of edges (children relationships)
fn get_stats(egraph: &EGraph<Mdl, TensorAnalysis>) -> (usize, usize, f32, usize) {
    let num_enodes = egraph.total_size();
    let num_classes = egraph.number_of_classes();
    let avg_nodes_per_class = num_enodes as f32 / (num_classes as f32);
    let num_edges = egraph.classes().fold(0, |acc, c| {
        c.iter().fold(0, |sum, n| n.len()+sum) + acc
    });
    (num_enodes, num_classes, avg_nodes_per_class, num_edges)
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
