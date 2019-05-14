#!/bin/bash --login

export MPLBACKEND="agg"

# flags:
## add flag --cuda to run script on GPUs, otherwise do not add flag

python3 main_gwgan_cnn.py --data fmnist --num_epochs 100 --beta 35
# (beta for MNIST: 32, for gray-scale CIFAR: 40)
