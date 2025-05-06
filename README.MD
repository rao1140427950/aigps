# Codes for "Artificial Intelligence-Generated Photonics: Map Optical Properties to Subwavelength Structures Directly via a Diffusion Model"

## Requirements:
```requirements.txt
jupyter==1.0.0
keras==2.12.0
matplotlib==3.7.1
numpy==1.23.5
protobuf==3.20.3
scipy==1.10.1
tensorflow==2.12.0
tqdm==4.65.0
```
The codes have been tested under Python 3.8 on Ubuntu 20.04 with RTX3080 GPU and CUDA 11.8.

## Overview
The inverse design method proposed in our paper has four neural network models: image encoder-decoder network, 
forward prediction networ, prompt encoder-decoder network and finally the latent diffusion network.

As image encoder-decoder network, forward prediction networ are prompt encoder-decoder network are all designed to 
supplement the latent diffusion network, we package and export these three models as tf.Module class and save them
in _models/exported_models_ for explicity. We reserve the codes for building
and training these three models in _models/image_cvae/cores_, _models/property_predictor/cores_ and _models/power_encoder/cores_.

## Usage
### Preparation
- Install the requirements.
- Download the exported models from [Google Drive](https://drive.google.com/file/d/1qLwxVXBe-zBmzfMyQ39YiCfAE2-wF76K/view?usp=sharing), 
unzip the files, and put them in _models/exported_models_ directory.
- Download the pretrained diffusion model from [Google Drive](https://drive.google.com/file/d/1mhgrdezP1llmjR_SFtVyhYYwcZur7Imv/view?usp=sharing), 
and put the checkpoint file in _checkpoints_ directory.

### Image encoder-decoder network
Run the following command:
```shell
python models/image_cvae/config.py
```
Running results of one test example will be shown as bellow:

![image](/srcs/image_cvae.png)


### Forward prediction network
Run the following command:
```shell
python models/property_predictor/config.py
```
Running results of one test example will be shown as bellow:

![image](/srcs/property_predictor.png)


### Prompt encoder-decoder network
Run the following command:
```shell
python models/power_encoder/config.py
```
Running results of one test example will be shown as bellow:

![image](/srcs/power_encoder.png)


### Latent diffusion network
Run the following command:
```shell
python test_diffusion.py
```
Running results of one test example will be shown as bellow:
![image](/srcs/test_diffusion.png)

The running results of _test_diffusion.py_ can also be found in _test_diffusion.ipynb_