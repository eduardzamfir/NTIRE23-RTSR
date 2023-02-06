import os
import logging
import torch

from collections import OrderedDict

from arch import SRModel


def main():
    
    """
    BASIC SETTINGS
    """
    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    """
    LOAD MODEL
    """
    model_path = os.path.join(os.getcwd(), 'checkpoint.pth')
    model = SRModel()
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    
    """
    READ IMAGES
    """
    testsets = os.path.join(os.getcwd(), 'data')
    testset_L = 'DIV2K_test_LR'
    L_folder = os.path.join(testsets, testset_L)
    E_folder = os.path.join(testsets, testset_L+'_results')
    
    """
    SETUP METRICS
    """
    test_results = OrderedDict()
    test_results['runtime'] = []
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)