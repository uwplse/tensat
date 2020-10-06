node_limit=50000
time_limit=3600
ilp_time_sec=3600

models=(
    nasrnn
    nasneta
)

for model in "${models[@]}"; do
    for iter_multi in 1 2
    do
        cargo run --release -- -r converted.txt -t converted_multi.txt -u -s none --n_iter $iter_multi --no_order --ilp_time_sec $ilp_time_sec --no_cycle --iter_multi $iter_multi --n_sec $time_limit --n_nodes $node_limit --all_weight_only -e ilp -d $model -o tmp/"$model"_eff_"$iter_multi"_stats.txt --saturation_only
        cargo run --release -- -r converted.txt -t converted_multi.txt -u -s none --n_iter $iter_multi --no_order --ilp_time_sec $ilp_time_sec --no_cycle --iter_multi $iter_multi --n_sec $time_limit --n_nodes $node_limit --all_weight_only -e ilp -d $model -o tmp/"$model"_vanilla_"$iter_multi"_stats.txt --filter_before --saturation_only
    done
done
