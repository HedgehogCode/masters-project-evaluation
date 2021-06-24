Steps to get the PSRN and FSIM values

1. Clone https://github.com/siavashBigdeli/DMSP-TF2 into the folder `DMSP` (`git clone git@github.com:siavashBigdeli/DMSP-TF2.git DMSP`)
1. Create the conda environment and activate `conda env create -f environment.yml`
1. Run `python main.py`


In a container (for example on a kubernetes cluster):
1. Start `nvidia/driver:460.73.01-ubuntu20.04` pod
```
$ apt update && apt upgrade
$ apt install wget vim tmux git
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
$ bash ~/miniconda.sh -p $HOME/miniconda
```
- Login again to use conda
- Create files environment.yml and main.py by copying them from here
```
$ conda env create -f environment.yml
$ git clone https://github.com/siavashBigdeli/DMSP-TF2.git DMSP
$ tmux
$ conda activate dmsp
$ export PYTHONUNBUFFERED=1
$ export CUDA_CACHE_MAXSIZE=2147483648
$ python main.py | tee /abyss/home/msganp-results/dmsp.out
```

- Detach from tmux with `<ctrl>-b`+`d`
- Attach to tmux with `tmux attach -t 0`
