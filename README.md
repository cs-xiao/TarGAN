# TarGAN

## Dependencies
* [Python 3.7+](https://www.python.org/downloads/)
* [PyTorch 1.6.0+](http://pytorch.org/)
* [Visualdl 2.0.5+](https://github.com/PaddlePaddle/VisualDL) (optional for log)
## Downloading datasets
To download the CHAOS dataset from [the CHAOS challenge website](https://chaos.grand-challenge.org/Download/). 
Then, you need to create a folder structure like this:

    datasets/chaos2019/
    ├── train
    │   ├── ct
    │   │   ├──1
    │   │   ├──...(patient index)
    │   ├── t1
    │   │   ├──3
    │   │   ├──...(patient index)
    │   ├── t2
    │   │   ├──5
    │   │   ├──...(patient index)
    ├── test
    │   ├── ct
    │   │   ├──2
    │   │   ├──...(patient index)
    │   ├── t1
    │   │   ├──4
    │   │   ├──...(patient index)
    │   ├── t2
    │   │   ├──6
    │   │   ├──...(patient index)
## Training networks
To train TarGAN on CHAOS, run the training script below. 

```bash
# Train TarGAN using the CHAOS dataset
python train.py -datasets chaos -save_path yours -epoch 50\
               -c_dim 3 -batch_size 4 -lr 1e-4 -ttur 3e-4\
               -random_seed 666
```
