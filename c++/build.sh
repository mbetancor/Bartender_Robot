#!/bin/bash


DIR=$(dirname $(realpath $0))
cd $DIR
cd build
cmake ..
make

if [[ "$1" == "-r" ]]; then
    ./ht
fi
