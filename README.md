## [NTIRE 2023 Workshop and Challenge](https://cvlai.net/ntire/2023/) @ CVPR 2023
### Real-Time Super-Resolution Challenge


**Prepare Test Dataset**

We degrade the high-resolution images of the DIV2K validation set with bicubic downsampling and JPEG compression. You can generate the low-resolution counterparts using following command.

````
python data/prepare_data.py --image-dir [IMAGE-ROOT] --lr-out-dir [LR-OUT-ROOT] --gt-out-dir [GT-OUT-DIR] --downsample-factor 4 --jpeg-level 90
````

**Test SR Model**

We run this file to generate SR outputs using your method. We save the outputs and compute the metrics offline.
````
python test.py --lr-dir [LR-ROOT] --save-dir [RESULTS-ROOT] --submission-id [SUBMISSION-ID] --checkpoint [CHECKPOINT] --scale [SCALE] --batch-size [BATCH-SIZE] --num-workers [NUM-WORKERS]
````

We run this file to calculate PSRN/SSIM (RGB and Y-Channel) metrics.
````
python calc_metrics.py --gt-dir [GT-ROOT] --save-dir [RESULTS-ROOT] --submission-id [SUBMISSION-ID]
````
