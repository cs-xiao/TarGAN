# TarGAN
It will be updated soon~
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

# Test TarGAN using the CHAOS dataset

```

## Using pre-trained networks
To download a pre-trained model checkpoint, run the script below. The pre-trained model checkpoint will be downloaded and saved into `./pre-trained/models` directory.

```bash

```
