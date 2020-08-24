num_passes=1
node_limit=50000
iter_limit=15
time_limit=300
iter_multi=2

for pass in $(seq 0 $(expr $num_passes - 1))
do
    model=resnext50
    cargo run -- -r converted.txt -t converted_multi.txt -u -s none --n_iter $iter_limit --no_order --ilp_time_sec 0 --no_cycle --iter_multi $iter_multi --n_sec $time_limit --n_nodes $node_limit --all_weight_only -e ilp -d $model -o tmp/"$model"_"$iter_multi"_stats.txt
done

for pass in $(seq 0 $(expr $num_passes - 1))
do
    model=bert
    cargo run -- -r converted.txt -t converted_multi.txt -u -s none --n_iter $iter_limit --no_order --ilp_time_sec 0 --no_cycle --iter_multi $iter_multi --n_sec $time_limit --n_nodes $node_limit --all_weight_only -e ilp -d $model -o tmp/"$model"_"$iter_multi"_stats.txt
done

for pass in $(seq 0 $(expr $num_passes - 1))
do
    model=nasrnn
    cargo run -- -r converted.txt -t converted_multi.txt -u -s none --n_iter $iter_limit --no_order --ilp_time_sec 0 --no_cycle --iter_multi $iter_multi --n_sec $time_limit --n_nodes $node_limit --all_weight_only -e ilp -d $model -o tmp/"$model"_"$iter_multi"_stats.txt
done

for pass in $(seq 0 $(expr $num_passes - 1))
do
    model=nasneta
    cargo run -- -r converted.txt -t converted_multi.txt -u -s none --n_iter $iter_limit --no_order --ilp_time_sec 0 --no_cycle --iter_multi $iter_multi --n_sec $time_limit --n_nodes $node_limit --all_weight_only -e ilp -d $model -o tmp/"$model"_"$iter_multi"_stats.txt
done