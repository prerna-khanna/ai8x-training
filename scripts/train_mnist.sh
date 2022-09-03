#!/bin/sh
#python train.py --gpus 1 --lr 0.1 --optimizer SGD --epochs 1 --deterministic --compress policies/schedule.yaml --model ai85netextrasmall_custom --dataset MNIST --confusion --param-hist --pr-curves --embedding --device MAX78000 "$@"

python train.py --gpus 1 --lr 0.1 --optimizer SGD --epochs 1 --deterministic --compress policies/schedule.yaml --model ai85netextrasmall_workload --dataset MNIST  --param-hist --pr-curves --embedding --device MAX78000 "$@"
