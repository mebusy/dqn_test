#!/bin/sh

# get the first parameter
script=$1

if [ -z "$script" ]; then
    echo "exectute dqn.py"
    script="dqn.py"
fi

set -e 

docker run --name gym --rm -it -v `pwd`:/opt/work/ \
    -e DISPLAY=$DISPLAY \
    mebusy/gym $script

