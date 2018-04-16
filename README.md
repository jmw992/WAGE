# Training and Inference with Integers in Deep Neural Networks - Ablation Study

Code example for the ICLR 2018 oral paper
I've taken the code commented it up a bit more, made it python 3 compatible, added some more options, and linked the options 

## Prerequisites
- NVIDIA GPU + CUDA + CuDNN
- Tensorflow (GPU version)
- python 3
- tqdm


## Data
Download and generate CIFAR10 dataset: 
```bash
cd dataSet/
python CIFAR10.py
```

## Config
Change your configurations in the file
```bash
gedit source/Option.py
```
## Train
Start training:
```bash
cd source/
python Top.py
```


