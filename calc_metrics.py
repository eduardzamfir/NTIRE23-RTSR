import os
import torch
import logging
import argparse

from tqdm import tqdm
from collections import OrderedDict


from arch import srmodel
from utils import util_logger
from utils import util_image as util
from utils.model_summary import get_model_flops



def main(args):
    """
    SETUP LOGGER
    """
    util_logger.logger_info("NTIRE2023-Real-Time-SR", log_path=os.path.join(args.save_dir, args.submission_id, f"Submission_{args.submission_id}.txt"))
    logger = logging.getLogger("NTIRE2023-Real-Time-SR")
    
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
    sr_images = util.get_image_paths(os.path.join(args.save_dir, args.submission_id, "results"))
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
        
        # transform images to Y channel
        sr_img_y = util.rgb2ycbcr(sr_img, only_y=True)
        hr_img_y = util.rgb2ycbcr(hr_img, only_y=True)
        
        # compute metrics
        test_results["psnr_rgb"].append(util.calculate_psnr(sr_img, hr_img, border=0))
        test_results["ssim_rgb"].append(util.calculate_ssim(sr_img, hr_img, border=0))
        test_results["psnr_y"].append(util.calculate_psnr(sr_img_y, hr_img_y, border=0))
        test_results["ssim_y"].append(util.calculate_ssim(sr_img_y, hr_img_y, border=0))


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
    parser.add_argument("--gt-dir", type=str)
    parser.add_argument("--save-dir", type=str, default="./outputs")
    parser.add_argument("--submission-id", type=str, default="1234")
    args = parser.parse_args()
    
    main(args)