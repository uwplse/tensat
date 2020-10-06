node_limit=50000
iter_limit=15
time_limit=3600
iter_multi=2
ilp_time_sec=3600

model=bert

for iter_multi in 3
do
    cargo run --release -- -r converted.txt -t converted_multi.txt -u -s none --n_iter $iter_multi --no_order --ilp_time_sec $ilp_time_sec --no_cycle --iter_multi $iter_multi --n_sec $time_limit --n_nodes $node_limit --all_weight_only -e ilp -d $model -o tmp/"$model"_vanilla_"$iter_multi"_stats.txt --filter_before --saturation_only
done

cargo run --release -- -r converted.txt -t converted_multi.txt -u -s none --n_iter 3 --no_order --ilp_time_sec 3600 --no_cycle --iter_multi 3 --n_sec 3600 --n_nodes 50000 --all_weight_only -e ilp -d bert -o tmp/bert_vanilla_3_stats.txt --filter_before --saturation_only
