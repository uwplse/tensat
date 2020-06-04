use crate::{model::*, rewrites::*};
use egg::*;

pub fn optimize(e: &RecExpr<Model>) -> RecExpr<Model> {
    let runner = Runner::default().with_expr(e).run(&rules());
    let (egraph, root) = (runner.egraph, runner.roots[0]);
    let mut extractor = Extractor::new(&egraph, Cost);
    extractor.find_best(root).1
}

struct Cost;
impl CostFunction<Model> for Cost {
    type Cost = (f64, Vec<usize>);
    fn cost<C: FnMut(Id) -> Self::Cost>(&mut self, enode: &Model, mut costs: C) -> Self::Cost {
        let children_sizes = enode.fold(vec![], |mut sizes, id| {
            sizes.push(costs(id).1);
            sizes
        });
        layouts(enode)
            .into_iter()
            .map(|layout| Self::run_time(enode, layout, &children_sizes))
            .min_by(|(x, _), (y, _)| x.partial_cmp(y).unwrap())
            .unwrap()
        // todo gotta calc output sizes
    }
}

struct Layout;
fn layouts(_e: &Model) -> Vec<Layout> {
    todo!()
}

impl Cost {
    fn run_time(
        _e: &Model,
        _layout: Layout,
        _sizes: &[Vec<usize>],
    ) -> <Cost as CostFunction<Model>>::Cost {
        todo!()
    }
}
