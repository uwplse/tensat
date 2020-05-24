use egg::{rewrite as rw, *};

define_language! {
    pub enum Model {
        "ewadd" = EAdd([Id; 2]),
        "ewmul" = EMul([Id; 2]),
        "smul" = SMul([Id; 2]),
        "transpose" = Trans(Id),
        "matmul" = MMul([Id; 2]),
        "conv" = Conv([Id; 5]),
        "enlarge" = Nlrg([Id; 2]),
        "relu" = Relu(Id),
        "poolavg" = Pola([Id; 4]),
        "poolmax" = Polm([Id; 4]),
        "concat" = Conc([Id; 3]),
        "split" = Splt([Id; 3]),
        "cpool" = Cpol,
        "iconv" = Icnv,
        "imatmul" = Imml,
        "iewmul" = Iewm,
        Num(i32),
    }
}

fn main() {
    println!("Hello, world!");
}
