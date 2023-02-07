import os
import pathlib
import argparse
from PIL import Image
from tqdm import tqdm


def main(args):
    
    pathlib.Path(os.path.join(args.lr_dir)).mkdir(parents=True, exist_ok=True)
    
    for filename in tqdm(os.listdir(args.gt_dir)):
        if filename.endswith(args.gt_file_ext):
            
            # load image
            img = Image.open(os.path.join(args.gt_dir, filename))
            img_name, ext = os.path.splitext(filename)
            
            # check sizes
            w, h = img.size
            assert w % args.downsample_factor == 0
            assert h % args.downsample_factor == 0

            # bicubic downsampling
            img = img.resize((int(w/args.downsample_factor), int(h/args.downsample_factor)), resample=Image.BICUBIC)
            
            # save to JPEG
            img.save(os.path.join(args.lr_dir, f"{img_name}.jpg"), "JPEG", quality=args.jpeg_level)
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir", type=str)
    parser.add_argument("--lr-dir", type=str)
    parser.add_argument("--jpeg-level", type=int, default=90)
    parser.add_argument("--downsample-factor", type=int, default=4)
    parser.add_argument("--gt-file-ext", type=str, default=".png")
    args = parser.parse_args()
    
    main(args)