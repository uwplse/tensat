num_passes=1
node_limit=10000
iter_limit=15
time_limit=3000
iter_multi=15
ilp_time_sec=3600
node_multi=3000

models=(
    inceptionv3
)

for model in "${models[@]}"; do
    for pass in $(seq 0 $(expr $num_passes - 1)); do
        cargo run -- -r converted.txt -t converted_multi.txt -u -s none --n_iter $iter_limit --no_order --ilp_time_sec 0 --no_cycle --iter_multi $iter_multi --n_sec $time_limit --n_nodes $node_limit --all_weight_only -e ilp -d $model -o tmp/"$model"_"$node_multi"_stats.txt --node_multi $node_multi #-x tmp/"$model"
    done
done
