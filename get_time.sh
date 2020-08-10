num_passes=5
for pass in $(seq 0 $(expr $num_passes - 1))
do
    cargo run -- -r converted.txt -t converted_multi.txt -u -s none --n_iter 15 -o bert_time.txt -e ilp -d bert
done