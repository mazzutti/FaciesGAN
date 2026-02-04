#!/usr/bin/env python3
import numpy as np

py_path = '/Users/mazzutti/POSDOC/Experimentos/FaciesGAN/results/py/2026_02_03_18_01_44/0/real_x_generated_facies/scale_0_epoch_0_fake.npy'
c_path = '/Users/mazzutti/POSDOC/Experimentos/FaciesGAN/results/c/2026_02_03_18_29_06/0/real_x_generated_facies/scale_0_epoch_99_fake.npy'

print("=== Python Fake Analysis ===")
py_data = np.load(py_path)
print(f"Shape: {py_data.shape}")
print(f"dtype: {py_data.dtype}")
print(f"Range: [{py_data.min():.4f}, {py_data.max():.4f}]")
print(f"Mean: {py_data.mean():.4f}, Std: {py_data.std():.4f}")
unique_vals = np.unique(py_data)
print(f"Exact unique values: {len(unique_vals)}")
print(f"Values: {unique_vals}")

print("\n=== C Fake Analysis ===")
c_data = np.load(c_path)
print(f"Shape: {c_data.shape}")
print(f"dtype: {c_data.dtype}")
print(f"Range: [{c_data.min():.4f}, {c_data.max():.4f}]")
print(f"Mean: {c_data.mean():.4f}, Std: {c_data.std():.4f}")
unique_c = np.unique(c_data.round(2))
print(f"Unique values (rounded): {len(unique_c)}")
print(f"First 10: {unique_c[:10]}")
