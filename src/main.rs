use tamago::{parse::*, verify::*};
use std::time::{Duration, Instant};
use tamago::model::*;
use tamago::rewrites::*;
use tamago::optimize::*;
use egg::*;
use std::env::*;
use std::fs::*;
use std::time::*;
use tamago::resnet50;

fn main() {
    //prove_taso_rules();
    //optimize();
    //convert_rw_rules();
    test();
}

fn convert_rw_rules() {
    env_logger::init();
    let file = args().nth(1).expect("Pls supply taso rules file.");
    let taso_rules = read_to_string(file).expect("Something went wrong reading the file");

    let converted = parse_and_convert(&taso_rules);

    write("converted.txt", converted).expect("Unable to write file");
}


fn test() {
    env_logger::init();
    
    let start = resnet50::get_resnet50();

    let runner_start = Runner::<Mdl, TensorAnalysis, ()>::default().with_expr(&start);
    runner_start.egraph.dot().to_svg("target/start.svg").unwrap();
}


/// Main procedure to run optimization
///
/// Reads input graph and rewrite rules from files specified as command line
/// arguments; runs saturation with TensorAnalysis dealing with metadata; runs
/// greedy extraction with TensorCost getting the cost per node/op; evaluates
/// full graph runtime of the starting graph and extracted graph.
fn optimize() {
    env_logger::init();

    // Reading input graph and rules
    /*
    let file = args().nth(1).expect("Pls supply example graph file.");
    let input_graph = read_to_string(file).expect("Something went wrong reading the file");
    let start = input_graph.parse().unwrap();
    */
    let start = resnet50::get_resnet50();

    let file_rules = args().nth(2).expect("Pls supply rewrite rules file.");
    let rw_rules = read_to_string(file_rules).expect("Something went wrong reading the rule file");
    let split_rules: Vec<&str> = rw_rules.split("\n").collect();

    /*use rand::seq::SliceRandom;
    let selected_rules = split_rules
        .choose_multiple(&mut rand::thread_rng(), 165).cloned()
        .collect();

    let rules = rules_from_str(selected_rules);*/
    let rules = rules_from_str(split_rules);

    // Run saturation
    let time_limit = Duration::new(100, 0);
    let iter_limit = 10;

    let start_time = Instant::now();
    let runner = Runner::<Mdl, TensorAnalysis, ()>::default()
        .with_time_limit(time_limit)
        .with_iter_limit(iter_limit)
        .with_expr(&start)
        .run(&rules[..]);
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
    let tnsr_cost = TensorCost {egraph: &egraph};
    let start_time = Instant::now();
    let mut extractor = Extractor::new(&egraph, tnsr_cost);
    let (best_cost, best) = extractor.find_best(root);
    let duration = start_time.elapsed();

    println!("Extractor complete!");
    println!("  Time taken: {:?}", duration);
    println!("  Best cost: {:?}", best_cost);

    // Evaluation starting and extracted graph runtime, save graphs
    let runner_ext = Runner::<Mdl, TensorAnalysis, ()>::default().with_expr(&best);
    runner_ext.egraph.dot().to_svg("target/ext.svg").unwrap();
    let time_ext = get_full_graph_runtime(&runner_ext);
    println!("Extracted graph runtime: {}", time_ext);

    let runner_start = Runner::<Mdl, TensorAnalysis, ()>::default().with_expr(&start);
    runner_start.egraph.dot().to_svg("target/start.svg").unwrap();
    let time_start = get_full_graph_runtime(&runner_start);
    println!("Start graph runtime: {}", time_start);
}


fn get_full_graph_runtime(runner: &Runner::<Mdl, TensorAnalysis, ()>) -> f32 {
    let mut g = runner.egraph.analysis.graph.borrow_mut();
    unsafe {
        g.run()
    }
}


fn prove_taso_rules() {
    env_logger::init();
    let file = args().nth(1).expect("Pls supply taso rules file.");
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
