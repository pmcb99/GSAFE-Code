import numpy as np
import sys
og = np.loadtxt('/DATA/features/Abuse/Abuse001_x264.txt')
og2 = np.load('/DATA/ReworkSultani/ogmini-gluon32segs/UCF/all_rgbs/Abuse/Abuse001_x264.mp4.npy')
print(og==og2)
print(f'Original Dims: {np.shape(og)}')
print(f'Original Dims: {np.shape(og2)}')