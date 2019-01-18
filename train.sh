#!/bin/bash
# source activate SENetPT

GPU=0

LOG_FILE=$(date +%Y%m%d_%H_%M_gpu$GPU.log)

python3 -m cirtorch.examples.train ckpt_2019_1_16 \
    --gpu-id '0'\
    --training-dataset 'retrieval-SfM-120k'\
    --test-datasets 'roxford5k' \
    --arch 'resnet101' \
    --pool 'gem' \
    --loss 'contrastive' \
    --loss-margin 0.85 \
    --optimizer 'adam' \
    --lr 5e-7 \
    --neg-num 5 \
    --query-size=2000 \
    --pool-size=22000 \
    --batch-size 5 \
    --image-size 362 2>&1| tee ${LOG_FILE} &

