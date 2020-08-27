node_limit=50000
iter_limit=15
time_limit=300
iter_multi=1
ilp_time_sec=3600

model=bert
cargo run -- -r converted.txt -t converted_multi.txt -u -s none --n_iter $iter_limit --ilp_time_sec $ilp_time_sec --iter_multi $iter_multi --n_sec $time_limit --n_nodes $node_limit --all_weight_only -e ilp -d $model -o tmp/"$model"_ilp_real_"$iter_multi"_stats.txt

cargo run -- -r converted.txt -t converted_multi.txt -u -s none --n_iter $iter_limit --ilp_time_sec $ilp_time_sec --iter_multi $iter_multi --n_sec $time_limit --n_nodes $node_limit --all_weight_only -e ilp -d $model -o tmp/"$model"_ilp_int_"$iter_multi"_stats.txt --order_var_int

iter_multi=2

cargo run -- -r converted.txt -t converted_multi.txt -u -s none --n_iter $iter_limit --ilp_time_sec $ilp_time_sec --iter_multi $iter_multi --n_sec $time_limit --n_nodes $node_limit --all_weight_only -e ilp -d $model -o tmp/"$model"_ilp_real_"$iter_multi"_stats.txt

cargo run -- -r converted.txt -t converted_multi.txt -u -s none --n_iter $iter_limit --ilp_time_sec $ilp_time_sec --iter_multi $iter_multi --n_sec $time_limit --n_nodes $node_limit --all_weight_only -e ilp -d $model -o tmp/"$model"_ilp_int_"$iter_multi"_stats.txt --order_var_int