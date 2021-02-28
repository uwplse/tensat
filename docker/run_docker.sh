docker run --gpus all --pid=host --net=host \
--name try \
-it \
--mount type=bind,source="/home/yichenyang/tensat",target=/usr/tensat \
--mount type=bind,source="/home/yichenyang/egg",target=/usr/egg \
--mount type=bind,source="/home/yichenyang/taso",target=/usr/TASO \
tensat:1.0 bash
