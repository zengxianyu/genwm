import sys
import pdb
import cv2
from tqdm import tqdm
sys.path.append("guided-diffusion")
from guided_diffusion.image_datasets import _list_image_files_recursively

path_in = sys.argv[1]
path_gt = sys.argv[2]

files_in = _list_image_files_recursively(path_in)

all_psnr = []
for file_in in tqdm(files_in):
    file_gt = file_in.replace(path_in, path_gt)
    img1 = cv2.imread(file_in)
    img2 = cv2.imread(file_gt)
    psnr = cv2.PSNR(img1, img2)
    all_psnr.append(psnr)
avg_psnr = sum(all_psnr)/float(len(all_psnr))
print(f"TotalPSNR: {avg_psnr}")
