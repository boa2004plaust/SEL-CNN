#!/bin/bash
# Build documentation for display in web browser.

LOG='log_CIR_MODEL'
SCENE=InF_DH
ARCH=mwtcnn
EPOCH=20
BS=500
LR=0.01


for SCENE in InF_DH indoor
do
    for ARCH in mwtcnn net4 net8 net12 net16
    do
        CUDA_VISIBLE_DEVICES=3,2,1,0 python train1D.py --exp-dir $LOG --scene $SCENE --arch $ARCH --epoch $EPOCH --batch_size $BS --lr $LR
    done
done