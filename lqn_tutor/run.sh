#!/bin/bash

# get the first parameter
script=$1

if [ -z "$script" ]; then
    # echo "exectute dqn.py"
    script="lqn.py"
fi

set -e 

# whether os is dawwin
if [ "$(uname)" == "Darwin" ]; then
    docker run --name gym --rm -it -v `pwd`:/opt/work/ \
        -v `pwd`/../fctorch/utils.py:/opt/work/fcutils.py \
        -e DISPLAY=$DISPLAY \
        mebusy/gym $script
else
    docker run  --rm -it -v `pwd`:/opt/work/  --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864  \
        -v `pwd`/../fctorch/utils.py:/opt/work/fcutils.py \
        -e DISPLAY='10.192.89.36:0' \
        mebusy/nv_torch_gym  python $script
fi





