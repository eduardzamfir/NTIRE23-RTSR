import os
import torch
import pathlib
import logging
import argparse
import numpy as np

from tqdm import tqdm
from PIL import Image
from collections import OrderedDict


from utils import util_logger
from utils import util_image as util
from utils import util_metrics as metrics



def main(args):
    """
    SETUP LOGGER
    """
    util_logger.logger_info("NTIRE2023-RTSR", log_path=os.path.join(args.sr_dir, args.submission_id, "results", f"Submission_{args.submission_id}.txt"))
    logger = logging.getLogger("NTIRE2023-RTSR")
    
    """
    SETUP METRICS
    """
    test_results = OrderedDict()
    test_results["psnr_rgb"] = []
    test_results["psnr_y"] = []
    test_results["ssim_rgb"] = []
    test_results["ssim_y"] = []
    
    """
    SETUP DIRS
    """
    sr_images = util.get_image_paths(os.path.join(args.sr_dir, args.submission_id, "results", f"SR{args.scale}"))
    hr_images = util.get_image_paths(args.gt_dir)
    
    """
    EVALUATE
    """
    for (sr, hr) in tqdm(zip(sr_images, hr_images)):
        # get image names
        sr_img_name, _ = os.path.splitext(os.path.basename(sr))
        hr_img_name, _ = os.path.splitext(os.path.basename(hr))
        
        # check if sr and hr match
        assert sr_img_name == hr_img_name
        
        # load image
        sr_img = util.imread_uint(sr, n_channels=3)
        hr_img = util.imread_uint(hr, n_channels=3)
        
        assert sr_img.dtype == np.uint8
        assert hr_img.dtype == np.uint8
        
        # compute metrics
        test_results["psnr_rgb"].append(metrics.calculate_psnr(sr_img, hr_img, crop_border=0))
        test_results["ssim_rgb"].append(metrics.calculate_ssim(sr_img, hr_img, crop_border=0))
        test_results["psnr_y"].append(metrics.calculate_psnr(sr_img, hr_img, crop_border=0, test_y_channel=True))
        test_results["ssim_y"].append(metrics.calculate_ssim(sr_img, hr_img, crop_border=0, test_y_channel=True))

    logger.info(f"------> Results of X{args.scale}")
    ave_psnr_rgb = sum(test_results["psnr_rgb"]) / len(test_results["psnr_rgb"])
    logger.info('------> Average PSNR (RGB) of ({}) is : {:.6f} dB'.format(args.submission_id, ave_psnr_rgb))
    ave_ssim_rgb = sum(test_results["ssim_rgb"]) / len(test_results["ssim_rgb"])
    logger.info('------> Average SSIM (RGB) of ({}) is : {:.6f}'.format(args.submission_id, ave_ssim_rgb))
    ave_psnr_y = sum(test_results["psnr_y"]) / len(test_results["psnr_y"])
    logger.info('------> Average PSNR (Y) of ({}) is : {:.6f} dB'.format(args.submission_id, ave_psnr_y))
    ave_ssim_y = sum(test_results["ssim_y"]) / len(test_results["ssim_y"]) 
    logger.info('------> Average SSIM (Y) of ({}) is : {:.6f}'.format(args.submission_id, ave_ssim_y))

        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir", type=str, default=os.path.join(pathlib.Path.home(), "projects/ntire/NTIRE23-RTSR", "demo/testset/GT"))
    parser.add_argument("--sr-dir", type=str, default=os.path.join(pathlib.Path.home(), "projects/ntire/NTIRE23-RTSR", "demo/submissions"))
    parser.add_argument("--submission-id", type=str, default="bicubic")
    parser.add_argument("--scale", default=2, type=int)
    args = parser.parse_args()
    
    main(args)