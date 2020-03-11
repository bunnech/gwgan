# GW GAN
(still under construction)

PyTorch Code for reproducing key results of the paper [Learning Generative Models across Incomparable Spaces](www.soon.com) by Charlotte Bunne, David Alvarez-Melis, Andreas Krause and Stefanie Jegelka.

## Citation
If you use this code for your research, please cite our paper.
```
@inproceedings{bunne2019,
  title={{Learning Generative Models across Incomparable Spaces}},
  author={Bunne, Charlotte and Alvarez-Melis, David and Krause, Andreas and Jegelka, Stefanie},
  year={2019},
  booktitle={International Conference on Machine Learning (ICML)},
  volume={97},
}
```

## Installation
To reproduce the experiments, download the source code from the Git repository.
```
git clone https://github.com/bunnech/gw_gan
```
Known **dependencies**: Python (3.7.2), numpy (1.15.4), pandas (0.23.4), matplotlib (3.0.2), seaborn (0.9.0), torch (1.0.0), torchvision (0.2.1).

## Experiments
We provide the source code to run GW GAN on a 2D Gaussians dataset as well as on MNIST, fashion-MNIST and gray-scale CIFAR.

### on a 2D Gaussians Dataset
In order to reproduce experiments on 2D Gaussians, you can either run the bash script `run_gwgan_mlp.sh` with pre-defined settings. Alternatively, call
```
python3 main_gwgan_mlp.py --modes 4mode --num_iter 10000 --l1reg
```
with the following environment options:
* `--modes` defined the number of modes. Available options are `4mode`, `5mode`, and `8mode`. To generate samples in 2D from 3D data, choose `3d_4mode`
* `--num_iter` defines the number of training iterations (recommended: 10000)
* `--l1reg` is a flag which activates l1-regularization (see paper for details)
* `--advsy` is a flag which activates the adversary.
* `--id` for identification of the training run.

### on MNIST, fashion-MNIST and gray-scale CIFAR Dataset
In order to reproduce experiments on 2D Gaussians, you can either run the bash script `run_gwgan_cnn.sh` with pre-defined settings. Alternatively, call
```
python3 main_gwgan_cnn.py --data fmnist --num_epochs 100 --beta 35
```
with the following environment options:
* `--data` selects the dataset. Choose between `mnist`, `fmnist` and `cifar_gray`.
* `--num_epochs` defines the number of training epochs (default: 200)
* `--n_channels` defines the number of channels of the CNN architecture (default=1).
* `--beta` defines the parameter of the Procrustes-based orthogonal regularization (recommended for MNIST (mnist): 32, fashion MNIST (fmnist): 35, gray-scale CIFAR (cifar_gray): 40)
* `--cuda` is a flag to run the code on GPUs.
* `--id` for identification of the training run.

## Code Structure
`.optra/gromov_wasserstein.py` contains the implementation of the Gromov-Wasserstein discrepancy with the mofifications described in the paper.

`.model/` contains scripts to generate datasets (`data.py`), the computation of the loss and regularization approaches (`loss.py`) as well as the network architectures (`model_mlp.py` and `model_cnn.py`).

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.
