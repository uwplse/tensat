docker run --gpus all --pid=host --net=host \
--name test \
-it \
--mount type=bind,source="/home/yichenyang/tamago",target=/usr/tamago \
--mount type=bind,source="/home/yichenyang/egg",target=/usr/egg \
--mount type=bind,source="/home/yichenyang/taso",target=/usr/TASO \
tamago:1.0 bash
