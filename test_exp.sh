node_limit=50000
iter_limit=15
time_limit=3000
iter_multi=2
ilp_time_sec=3600

model=bert

cargo run --release -- -r converted.txt -t converted_multi.txt -u -s none --n_iter $iter_limit --no_order --ilp_time_sec $ilp_time_sec --no_cycle --iter_multi $iter_multi --n_sec $time_limit --n_nodes $node_limit --all_weight_only -e ilp -d $model -o tmp/"$model"_"$iter_multi"_stats.txt
