#!/bin/bash
# Build documentation for display in web browser.

LOG='log_CIR_MODEL'
SCENEFROM=InF_DH
SCENETO=indoor
ARCH=mwtcnn
EPOCH=20
BS=500
LR=0.01

for ARCH in net4 net8 net12 net16
do
    CUDA_VISIBLE_DEVICES=3,2,1,0 python test1D.py --exp-dir $LOG --sceneFrom $SCENEFROM --sceneTo $SCENETO --arch $ARCH --epoch $EPOCH --batch_size $BS --lr $LR
done