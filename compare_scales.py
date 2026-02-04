#!/usr/bin/env python3
import numpy as np
from PIL import Image
import sys

py_base = 'results/py/2026_02_03_17_46_32'
c_base = 'results/c/2026_02_03_19_34_06'

print('Scale | Py unique | C unique | Py pure% | C pure% | Mean diff | Max diff')
print('-' * 85)

pure_colors = {(0,0,0), (255,0,0), (0,255,0), (0,0,255)}

for scale in range(7):
    py_path = f'{py_base}/{scale}/real_x_generated_facies/gen_{scale}_99.png'
    c_path = f'{c_base}/{scale}/real_x_generated_facies/gen_{scale}_99.png'
    
    py_img = np.array(Image.open(py_path))
    c_img = np.array(Image.open(c_path))
    
    py_flat = py_img.reshape(-1, 3)
    c_flat = c_img.reshape(-1, 3)
    
    py_colors = set(map(tuple, py_flat))
    c_colors = set(map(tuple, c_flat))
    
    diff = np.abs(py_img.astype(float) - c_img.astype(float))
    mean_diff = diff.mean()
    max_diff = diff.max()
    
    py_pure_count = sum(1 for px in py_flat if tuple(px) in pure_colors)
    py_pure_pct = 100.0 * py_pure_count / len(py_flat)
    
    c_pure_count = sum(1 for px in c_flat if tuple(px) in pure_colors)
    c_pure_pct = 100.0 * c_pure_count / len(c_flat)
    
    print(f'{scale:5} | {len(py_colors):9} | {len(c_colors):8} | {py_pure_pct:7.1f}% | {c_pure_pct:6.1f}% | {mean_diff:9.2f} | {max_diff:8.0f}')
