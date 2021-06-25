docker run --gpus all --pid=host --net=host \
--name try \
-it \
--mount type=bind,source="$(pwd)",target=/usr/tensat \
tensat:1.0 bash
