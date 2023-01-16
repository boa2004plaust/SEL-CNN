#!/bin/bash
# Build documentation for display in web browser.

LOG='log_CIR'
SCENE=MWT
ARCH=mwtcnn2d
EPOCH=10
BS=64
LR=0.0003


for ARCH in mwtcnn2d
do
    CUDA_VISIBLE_DEVICES=1 python train2D_MWTCNN.py \
                --exp-dir $LOG \
                --scene $SCENE \
                --arch $ARCH \
                --epoch $EPOCH \
                --batch_size $BS \
                --lr $LR
done


LOG='log_CIR'
SCENE=MWTfull
ARCH=mwtcnn2d
EPOCH=10
BS=64
LR=0.0003


for ARCH in mwtcnn2d
do
    CUDA_VISIBLE_DEVICES=1 python train2D_MWTCNN.py \
                --exp-dir $LOG \
                --scene $SCENE \
                --arch $ARCH \
                --epoch $EPOCH \
                --batch_size $BS \
                --lr $LR
done