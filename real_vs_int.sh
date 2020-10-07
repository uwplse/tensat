node_limit=50000
iter_limit=15
time_limit=3000
iter_multi=1
ilp_time_sec=3600

models=(
    nasrnn
    nasneta
    inceptionv3
    resnext50
    bert
)

for model in "${models[@]}"; do
    for iter_multi in 1 2
    do
        cargo run --release -- -r converted.txt -t converted_multi.txt -u -s none --n_iter $iter_limit --ilp_time_sec $ilp_time_sec --iter_multi $iter_multi --n_sec $time_limit --n_nodes $node_limit --all_weight_only -e ilp -d $model -o tmp/"$model"_ilp_real_"$iter_multi"_stats.txt

        cargo run --release -- -r converted.txt -t converted_multi.txt -u -s none --n_iter $iter_limit --ilp_time_sec $ilp_time_sec --iter_multi $iter_multi --n_sec $time_limit --n_nodes $node_limit --all_weight_only -e ilp -d $model -o tmp/"$model"_ilp_int_"$iter_multi"_stats.txt --order_var_int
    done
done
