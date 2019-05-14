#!/bin/bash --login

export MPLBACKEND="agg"

# flags:
## add flag --advsy to run script with adversary, otherwise do not add flag
## add flag --l1reg to run script with additional regularisation term, otherwise do not add flag

# python3 main_gwgan_mlp.py --modes 3d_4mode --num_iter 10000
python3 main_gwgan_mlp.py --modes 4mode --num_iter 10000 --l1reg --advsy
