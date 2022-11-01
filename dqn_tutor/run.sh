#!/bin/sh

# get the first parameter
script=$1

if [ -z "$script" ]; then
    echo "exectute dqn.py"
    script="dqn.py"
fi

set -e 

# docker run --name gym --rm -it -v `pwd`:/opt/work/ \
#     -e DISPLAY=$DISPLAY \
#     mebusy/gym $script


docker run  --rm -it -v `pwd`:/opt/work/  --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864  \
    -e DISPLAY='10.192.89.36:0' \
    mebusy/nv_torch_gym  python $script


