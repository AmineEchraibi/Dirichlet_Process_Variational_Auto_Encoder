# Dirichlet Process Variational AutoEncoder
In this repository you will find the implementation of the DP-VAE described in the following paper: https://hal.archives-ouvertes.fr/hal-02864385v2
## Graphical model

<img src="results/model1.png" alt="drawing" width="400"/>

## Mathematical model

![alt text](results/model2.png =100x20)

## Generated samples

![alt text](results/result.png =100x20e)

## Training code

```console
foo@bar:~$ python train.py --mnistdir dir 
```
```console
foo@bar:~$ tensorboard --logdir runs
```
