import os
import numpy as np
from PIL import Image

def convert_to_yuv(png_dir, output_yuv_file, width=3840, height=2160, seq_len=129):
    with open(output_yuv_file, "wb") as f_y:
        for frame_num in range(seq_len):
            image_path = os.path.join(png_dir, f'{frame_num:03d}.png')
            image = Image.open(image_path)
            if image.mode != 'L':
                image = image.convert('L')
            pixels = np.array(image)
            
            f_y.write(pixels.tobytes())

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--png_dir', type=str, required=True, help='Directory containing PNG images')
    parser.add_argument('-o', '--output_yuv_file', type=str, required=True, help='Output YUV file path')
    args = parser.parse_args()

    png_dir = args.png_dir
    output_yuv_file = args.output_yuv_file

    convert_to_yuv(png_dir, output_yuv_file)
