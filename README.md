## [NTIRE 2023 Workshop and Challenge](https://cvlai.net/ntire/2023/) @ CVPR 2023
## Real-Time Super-Resolution Challenge

## Prepare Test Dataset
We degrade the high-resolution images of the DIV2K validation set with bicubic downsampling and JPEG compression. You can generate the low-resolution counterparts using following command.

    ```
    python data/prepare_data.py --gt-dir [GT-ROOT] --lr-dir [LR-ROOT] --downsample-factor 4 --jpeg-level 90 --gt-file-ext .png
    ```