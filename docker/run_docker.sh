docker run --gpus all --pid=host --net=host \
--name devbegg \
-it \
--mount type=bind,source="/home/yicheny_google_com/tamago",target=/usr/tamago \
--mount type=bind,source="/home/yicheny_google_com/egg",target=/usr/egg \
tamago:local bash
#docker run --gpus all --pid=host --net=host -it tamago:0.1 bash
