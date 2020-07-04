use tamago::{parse::*, verify::*};
use std::time::{Duration, Instant};

fn main() {
    //prove_taso_rules();
    //optimize();
    //convert_rw_rules();
    test();
}

fn convert_rw_rules() {
    use std::env::*;
    use std::fs::*;

    env_logger::init();
    let file = args().nth(1).expect("Pls supply taso rules file.");
    let taso_rules = read_to_string(file).expect("Something went wrong reading the file");

    let converted = parse_and_convert(&taso_rules);

    write("converted.txt", converted).expect("Unable to write file");
}


fn test() {
    use tamago::model::*;
    use tamago::rewrites::*;
    use egg::*;
    use std::env::*;
    use std::fs::*;

    env_logger::init();
    let file = args().nth(1).expect("Pls supply example graph file.");
    let input_graph = read_to_string(file).expect("Something went wrong reading the file");
    let start = input_graph.parse().unwrap();

    let runner = Runner::<Mdl, TensorAnalysis, ()>::default()
        .with_expr(&start);

    println!("  Nodes: {}", runner.egraph.total_size());
}

fn optimize() {
    use tamago::model::*;
    use tamago::rewrites::*;
    use egg::*;
    use std::env::*;
    use std::fs::*;
    use std::time::*;

    env_logger::init();
    let file = args().nth(1).expect("Pls supply example graph file.");
    let input_graph = read_to_string(file).expect("Something went wrong reading the file");

    let file_rules = args().nth(2).expect("Pls supply rewrite rules file.");
    let rw_rules = read_to_string(file_rules).expect("Something went wrong reading the rule file");
    let split_rules: Vec<&str> = rw_rules.split("\n").collect();

    /*use rand::seq::SliceRandom;
    let selected_rules = split_rules
        .choose_multiple(&mut rand::thread_rng(), 165).cloned()
        .collect();

    let rules = rules_from_str(selected_rules);*/
    let rules = rules_from_str(split_rules);

    let ten_seconds = Duration::new(100, 0);

    let start = input_graph.parse().unwrap();
    let start_time = Instant::now();
    //let runner = Runner::<Mdl, (), ()>::default().with_expr(&start).run(&rules()[..]);
    let runner = Runner::<Mdl, TensorAnalysis, ()>::default()
        .with_time_limit(ten_seconds)
        .with_iter_limit(1000)
        .with_expr(&start)
        .run(&rules[..]);
    let duration = start_time.elapsed();

    println!("Runner complete!");
    println!("  Nodes: {}", runner.egraph.total_size());
    println!("  Classes: {}", runner.egraph.number_of_classes());
    println!("  Stopped: {:?}", runner.stop_reason.unwrap());
    println!("  Time taken: {:?}", duration);

    let (egraph, root) = (runner.egraph, runner.roots[0]);

    let mut extractor = Extractor::new(&egraph, AstSize);
    let start_time = Instant::now();
    let (best_cost, best) = extractor.find_best(root);
    let duration = start_time.elapsed();

    println!("Extractor complete!");
    println!("  Time taken: {:?}", duration);

    let runner_ext = Runner::<Mdl, (), ()>::default().with_expr(&best);
    runner_ext.egraph.dot().to_svg("target/ext.svg").unwrap();

    let runner_start = Runner::<Mdl, (), ()>::default().with_expr(&start);
    runner_start.egraph.dot().to_svg("target/start.svg").unwrap();

    //egraph.dot().to_svg("target/tamago.svg").unwrap();
}

fn prove_taso_rules() {
    use std::env::*;
    use std::fs::*;

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
