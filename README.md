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
python data/prepare_data.py --image-dir [IMAGE-ROOT] --lr-out-dir [LR-OUT-ROOT] --gt-out-dir [GT-OUT-DIR] --downsample-factor 4 --jpeg-level 90
````

**Test SR Model**

We run this file to generate SR outputs using your method and compute FLOPs and runtime. We save the outputs and compute the metrics offline.
````
python test.py --lr-dir [LR-ROOT] --save-dir [RESULTS-ROOT] --submission-id [SUBMISSION-ID] --checkpoint [CHECKPOINT] --scale [SCALE] --batch-size [BATCH-SIZE] --num-workers [NUM-WORKERS]
````

We run this file to calculate PSRN/SSIM (RGB and Y-Channel) metrics.
````
python calc_metrics.py --gt-dir [GT-ROOT] --save-dir [RESULTS-ROOT] --submission-id [SUBMISSION-ID]
````
