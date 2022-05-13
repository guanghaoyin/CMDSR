#!/bin/bash


THIS_DIR="$( cd "$( dirname "$0" )" && pwd )"
#echo $THIS_DIR
#mkdir -p /root/.cache/torch/checkpoints/
#cp /mnt/cephfs_new_wj/vc/sunjia.ly/super_resolution/models/vgg19-dcbb9e9d.pth /root/.cache/torch/checkpoints/vgg19-dcbb9e9d.pth
# export DOUBAN=https://pypi.doubanio.com/simple/

cd $THIS_DIR
python3 Train_CMDSR.py "$@"
# sleep 1d
