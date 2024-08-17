import h5py
import matplotlib.pyplot as plt
import numpy as np


file_path = r'C:\Users\lenovo\OneDrive\Desktop\mamba\data\0615data\test\low\data.h5'
with h5py.File(file_path, 'r') as h5_file:
    data = h5_file['data'][:]

# 检查数据形状
print("Data shape:", data.shape)


plt.imshow(data, cmap='gray', aspect='auto')
plt.colorbar()
plt.xlabel('Traces')
plt.ylabel('Sample')
plt.show()
