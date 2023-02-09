import os
import torch
import pathlib
import logging
import argparse

from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data import DataLoader

import data.dataset as dd
from arch import srmodel
from utils import util_logger
from utils import util_image as util
from utils.model_summary import get_model_flops



def main(args):
    """
    SETUP DIRS
    """
    pathlib.Path(os.path.join(args.save_dir, args.submission_id, "results")).mkdir(parents=True, exist_ok=True)
    
    """
    SETUP LOGGER
    """
    util_logger.logger_info("NTIRE2023-Real-Time-SR", log_path=os.path.join(args.save_dir, args.submission_id, f"Submission_{args.submission_id}.txt"))
    logger = logging.getLogger("NTIRE2023-Real-Time-SR")
    
    """
    BASIC SETTINGS
    """
    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
   
    """
    LOAD MODEL
    """
    model_path = os.path.join(args.checkpoint)
    model = srmodel(scale=args.scale)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    
    # number of parameters
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info('Params number: {}'.format(number_parameters))
    
    
    """
    SETUP DATALOADER
    """
    #transforms = dd.Compose([dd.CenterCrop(crop_size=args.crop_size)])
    dataset = dd.SRDataset(lr_images_dir=args.lr_dir, n_channels=args.n_channels, transform=None)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    
    """
    SETUP RUNTIME
    """
    test_results = OrderedDict()
    test_results["runtime"] = []

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    """
    TESTING
    """
    for img_L, img_path in tqdm(dataloader):
        # get LR image
        img_name, ext = os.path.splitext(img_path[0])

        # load LR image
        img_L = img_L.to(device)

        # forward pass
        start.record()
        img_E = model(img_L)
        end.record()
        torch.cuda.synchronize()
        test_results["runtime"].append(start.elapsed_time(end))  # milliseconds

        # postprocess
        img_E = util.tensor2uint(img_E)
        
        # save model output
        util.imsave(img_E, os.path.join(os.path.join(args.save_dir, args.submission_id, "results", img_name + ".png")))
    
    
    input_dim = (3, 256, 256)  # set the input dimension
    flops = get_model_flops(model, input_dim, False)
    flops = flops / 10 ** 9
    logger.info("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

    num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    num_parameters = num_parameters / 10 ** 6
    logger.info("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters))

    ave_runtime = sum(test_results["runtime"]) / len(test_results["runtime"]) / 1000.0
    logger.info('------> Average runtime of ({}) is : {:.6f} seconds'.format(args.submission_id, ave_runtime))
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr-dir", type=str, default="/home/eduardzamfir/datasets/image_restoration/NTIRE23SR/LR/3x/Test")
    parser.add_argument("--save-dir", type=str, default="./outputs")
    parser.add_argument("--submission-id", type=str, default="1234")
    parser.add_argument("--n-channels", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pth")
    parser.add_argument("--crop-size", type=int, nargs="+", default=[2048, 1080])
    parser.add_argument("--scale", type=int, default=3)
    args = parser.parse_args()
        
    main(args)