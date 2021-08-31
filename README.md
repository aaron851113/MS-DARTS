# MS-DARTS
paper : MS-DARTS: Mean-Shift Based Differentiable Architecture Search [link](https://arxiv.org/abs/2108.09996) <br />
This code is based on the implementation of [DARTS](https://github.com/quark0/darts), [NAS-Bench-1Shot1](https://github.com/automl/nasbench-1shot1) and [SmoothDARTS](https://github.com/xiangning-chen/SmoothDARTS/tree/e8d80c3c1f22f596a8b38808b34ad9a2f833bb9d).

### Download 
- nasbench tfrecord [link](https://drive.google.com/file/d/1tzXIHs-H3qGAfJEzhBpT5BeBZICHXIGL/view?usp=sharing)
- Imagenet weight [link](https://drive.google.com/file/d/1aDEx0AdCrugkTEgUnbo6f6Zf4Vz2M1dr/view?usp=sharing)
- Cifar10 weight [link](https://drive.google.com/file/d/1rk893INNIxUxZk0ftVQN3Wf7PDdeUmHK/view?usp=sharing)

### training search
`$ cd MS-DARTS/sota/cnn` <br />
`$ python train_search_ms.py`

### evluation
- Get the architecture with best validation accuracy in training search stage
- Add the architecture in genotypes.py
- train these Genotype with following: <br />
`$ cd MS-DARTS/sota/cnn` <br />
`$ python train.py`
