#!/usr/bin/env python3
from PIL import Image
import numpy as np

# Compare PNG outputs
py_png = '/Users/mazzutti/POSDOC/Experimentos/FaciesGAN/results/py/2026_02_03_17_46_32/6/real_x_generated_facies/gen_6_99.png'
c_png = '/Users/mazzutti/POSDOC/Experimentos/FaciesGAN/results/c/2026_02_03_18_29_06/6/real_x_generated_facies/gen_6_99.png'

py_img = np.array(Image.open(py_png))
c_img = np.array(Image.open(c_png))

# Check mean colors
print('=== Color Distribution ===')
print('Python image mean RGB:', py_img.mean(axis=(0,1)))
print('C image mean RGB:', c_img.mean(axis=(0,1)))

# Color histogram comparison  
print('\n=== Pure color check (checking for facies-like colors) ===')
# Expected pure colors in RGB [0,255]: black(0,0,0), red(255,0,0), green(0,255,0), blue(0,0,255)
for name, img in [('Python', py_img), ('C', c_img)]:
    # Reshape to get all pixels
    pixels = img.reshape(-1, 3)
    # Count approximate pure colors
    black = np.sum(np.all(pixels < 30, axis=1))
    red = np.sum((pixels[:,0] > 200) & (pixels[:,1] < 50) & (pixels[:,2] < 50))
    green = np.sum((pixels[:,0] < 50) & (pixels[:,1] > 200) & (pixels[:,2] < 50))
    blue = np.sum((pixels[:,0] < 50) & (pixels[:,1] < 50) & (pixels[:,2] > 200))
    total = pixels.shape[0]
    print(f'{name}: black={black/total*100:.1f}%, red={red/total*100:.1f}%, green={green/total*100:.1f}%, blue={blue/total*100:.1f}%')

# Check if images look similar by computing SSIM-like metrics
print('\n=== Difference Analysis ===')
diff = np.abs(py_img.astype(float) - c_img.astype(float))
print(f'Mean absolute difference: {diff.mean():.2f}')
print(f'Max absolute difference: {diff.max():.2f}')
