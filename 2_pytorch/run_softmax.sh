#!/bin/sh
#############################################################################
# TODO: Initialize anything you need for the forward pass
#############################################################################
python -u train.py \
    --model softmax \
    --epochs 20 \
    --weight-decay 0.001 \
    --momentum 0.95 \
    --batch-size 256 \
    --lr 0.001 | tee softmax.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
