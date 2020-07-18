docker run --gpus all --pid=host --net=host \
--name devball \
-it \
--mount type=bind,source="/home/yicheny_google_com/tamago",target=/usr/tamago \
--mount type=bind,source="/home/yicheny_google_com/egg",target=/usr/egg \
--mount type=bind,source="/home/yicheny_google_com/taso",target=/usr/TASO \
tamago:ball bash