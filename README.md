# [NTIRE 2023 Real-Time Super-Resolution Challenge](https://cvlai.net/ntire/2023/) @ CVPR 2023

## About the Challenge
The [8th edition of NTIRE: New Trends in Image Restoration and Enhancement](https://cvlai.net/ntire/2023/) workshop will be held on June 18th, 2023 in conjunction with CVPR 2023.

Image Super-Resolution is one of the most popular computer vision problems due to its real-world applications: photography, gaming, generative AI, etc. The goal of the NTIRE 2023 Real-Time Super-Resolution Challenge is to upscale images in real-time at 60FPS using deep learning models and commercial GPUs (RTX 3060, 3090). 
The input images can be large patches or full-resolution images, compressed using JPEG q=90. The challenge has two tracks:

**Track 1**: Upscaling from FHD 1080p to 4K resolution (X2 factor) | [CodaLab Server](https://codalab.lisn.upsaclay.fr/competitions/10227)

**Track 2**: Upscaling from HD 720p to 4K resolution (X3 factor) | [CodaLab Server](https://codalab.lisn.upsaclay.fr/competitions/10228)

The submitted methods will be tested to ensure they satisfy real-time processing on RTX 3060 (12Gb) / RTX 3090 (24Gb), and will be ranked based on the fidelity (PSNR, SSIM) of their results w.r.t. the high-resolution reference images in our internal test set. The scoring kernel will be provided to the participants. More details about the specs of the VM we use to run the models and measure metrics such as FLOPs, memory consumption, runtime per image (ms), will be provided to the participants via GitHub.

**IMPORTANT**

* Participants can train the models using **any** publicly available open-sourced dataset. Although, complete details must be provided in the report.
* The validation/test dataset consists on a brand-new dataset that includes high-quality filtered content from: **digital art**, **videogames**, **photographies** - Please consider this variety when training your models.

## About this repository
**Prepare Test Dataset**

We degrade the high-resolution images with bicubic downsampling and JPEG compression. You can generate the low-resolution counterparts using following command.

````
python demo/data/prepare_data.py --image-dir [IMAGE-ROOT] --lr-out-dir [LR-OUT-ROOT] --gt-out-dir [GT-OUT-DIR] --downsample-factor [2|3] --jpeg-level 90
````

## **Evaluation of your submission**

We request that you submit a ```submission_{submission-id}.zip``` file, which should include the following components:

```
submission_{submission-id}.zip/
|--- arch.py
|--- utils/
|    |--- modules.py
|    |--- config.yaml
|    ...
|--- checkpoint.pth
|--- results/
|    |--- 1.png
|    |--- 2.png
|    ...
|--- requirements.txt
```

* ```arch.py```: This file contains your network architecture. Additionally, we request a simple ```srmodel()``` method which returns an instance of your method initialized from your submitted ```checkpoint.pth``` file. In case you are submitting multiple checkpoints, we select a single file randomly. 
* ```utils/```: You may have an additional directory ```utils/``` containing necessary scripts and files to run your model. Please be aware that we expect ```srmodel()``` to return your method with correct configuration, checkpoint etc. **without** input arguments.
* ```results/```: This directory contains your SR outputs saved as ```.png``` files. We calculate PSNR/SSIM metrics using your provided super-resolved images and compare to our internal evaluation of your method using ```test.py```.
* ```requirements.txt```: Please provide an ```requirements.txt``` file in case you use additional libraries besides the ones described in **our** ```requirements.txt``` file.

### Evalutation procedure

We compute our metrics using ```calc_metrics.py``` and the SR outputs you provide in ```results/```. Please ensure that you adhere to our naming conventions. We report average PSNR/SSIM on RGB and Y-Channel.
```
python calc_metrics.py --submission-id [YOUR-SUBMISSION-ID] --sr-dir ./results --gt-dir [PATH-TO-OUR-GT]
```
Next, we use ```test.py``` to compute the super-resolved outputs of your submitted method. The SR images will be saved to ```internal/```. Besides, we compute the average runtime of your model per image and report FLOPs.
```
python test.py --submission-id [YOUR-SUBMISSION-ID] --checkpoint [PATH-TO-YOUR-CHECKPOINT] --scale [2|3] --lr-dir [PATH-TO-OUR-LR]
``` 
