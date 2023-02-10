import os
import pathlib
import argparse
from PIL import Image
from tqdm import tqdm
from torchvision.transforms.functional import center_crop

def main(args):
    
    pathlib.Path(os.path.join(args.lr_out_dir)).mkdir(parents=True, exist_ok=True)
    
    if args.gt_out_dir is not None:
        pathlib.Path(os.path.join(args.gt_out_dir)).mkdir(parents=True, exist_ok=True)
    
    for filename in tqdm(os.listdir(args.image_dir)):
        # load image
        img = Image.open(os.path.join(args.image_dir, filename)).convert('RGB')
        img_name, ext = os.path.splitext(filename)
              
        # pre-crop image to DIV2K dimensions
        img = center_crop(img, output_size=args.crop_size)
        if args.gt_out_dir is not None:
            img.save(os.path.join(args.gt_out_dir, f"{img_name+ext}"))

        # check sizes
        w, h = img.size
        assert w % args.downsample_factor == 0
        assert h % args.downsample_factor == 0
        
        # bicubic downsampling
        img = img.resize((int(w/args.downsample_factor), int(h/args.downsample_factor)), resample=Image.Resampling.BICUBIC)
        
        if ext == ".jpg":
            img.save(os.path.join(args.lr_out_dir, f"{img_name}.jpg"), "JPEG", quality=100)
        else:
            # save to JPEG
            img.save(os.path.join(args.lr_out_dir, f"{img_name}.jpg"), "JPEG", quality=args.jpeg_level)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=str)
    parser.add_argument("--gt-out-dir", type=str, default=None)
    parser.add_argument("--lr-out-dir", type=str)
    parser.add_argument("--jpeg-level", type=int, default=90)
    parser.add_argument("--downsample-factor", type=int, default=2)
    parser.add_argument("--crop-size", type=int, default=[1080, 2040], nargs="+")
    args = parser.parse_args()
    
    main(args)