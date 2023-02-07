import os
import torch
import pathlib
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
    model = srmodel()
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    
    # number of parameters
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info('Params number: {}'.format(number_parameters))
    
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
    for img in tqdm(util.get_image_paths(args.lr_dir)):
        # get LR image
        img_name, ext = os.path.splitext(os.path.basename(img))

        # load LR image
        img_L = util.imread_uint(img, n_channels=3)
        img_L = util.uint2tensor4(img_L)
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
    parser.add_argument("--lr-dir", type=str, default="./testset")
    parser.add_argument("--save-dir", type=str, default="./outputs")
    parser.add_argument("--submission-id", type=str, default="1234")
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pth")
    args = parser.parse_args()
        
    main(args)