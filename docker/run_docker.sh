docker run --gpus all --pid=host --net=host \
--name try \
-it \
--mount type=bind,source="/home/remywang/tamago",target=/usr/tamago \
--mount type=bind,source="/home/remywang/egg",target=/usr/egg \
--mount type=bind,source="/home/remywang/taso",target=/usr/TASO \
tamago:1.0 bash
