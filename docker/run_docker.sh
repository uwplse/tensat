docker run --gpus all --pid=host --net=host \
--name dv \
-it \
--mount type=bind,source="/home/yicheny_google_com/tamago",target=/usr/tamago \
tamago:mnt bash
#docker run --gpus all --pid=host --net=host -it tamago:0.1 bash
