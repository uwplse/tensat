use tamago::{parse::*, verify::*};

fn main() {
  use tamago::model::*;
  use tamago::rewrites::*;
  use egg::*;
  
  let start = "(input 0 0)".parse().unwrap();
  let runner = Runner::<Mdl, TensorAnalysis, ()>::default().with_expr(&start).run(&rules()[..]);
  runner.egraph.dot().to_svg("target/tamago.svg").unwrap();
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
