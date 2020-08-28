docker run --gpus all --pid=host --net=host \
--name test \
-it \
--mount type=bind,source="/home/yicheny_google_com/tamago",target=/usr/tamago \
--mount type=bind,source="/home/yicheny_google_com/egg",target=/usr/egg \
--mount type=bind,source="/home/yicheny_google_com/taso",target=/usr/TASO \
tamago:1.0 bash