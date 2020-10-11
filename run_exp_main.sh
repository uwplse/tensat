node_limit=50000
iter_limit=15
time_limit=3600
ilp_time_sec=3600

num_passes=5

models=(
    inceptionv3
    resnext50
    bert
    nasrnn
    nasneta
    squeezenet
    vgg
)

for pass in $(seq 0 $(expr $num_passes - 1)); do
    for iter_multi in 0 1 2; do
        for model in "${models[@]}"; do
            cargo run --release -- -r converted.txt -t converted_multi.txt -u -s none --n_iter $iter_limit --no_order --ilp_time_sec 0 --no_cycle --iter_multi $iter_multi --n_sec $time_limit --n_nodes $node_limit --all_weight_only -e ilp -d $model -o tmp/"$model"_"$iter_multi"_stats.txt # -x tmp/"$model"_optimized.model
        done
    done
done
