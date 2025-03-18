from process.d_trainfunction import train, test_normal
# import matplotlib
# matplotlib.use('tkagg')
import random
import numpy as np
import torch
datasets_paths = [
# "data/esc10-all/npy/coc_out",
"data/us8k-all/npy/coc_out",
# "data/pump-all/npy/coc_out",
# "data/engine-all/npy/coc_out",
]

# train(datasets_paths, path2tab='tab', path2trained='bestmodel',
#       trials=2, batch_size=5, num_epochs=2, initial_lr=0.1, step_size=5, test_train=True)
#
train(datasets_paths, path2tab='tab', path2trained='bestmodel', method='cgss',
      trials=1, batch_size=32, num_epochs=110, initial_lr=0.1, step_size=100)  # 20-18 pump 8-6 engine 150-110 us8k


test_normal(datasets_paths[0], trials=5, batch_size=32, method='cgss')

# test_ensemble(datasets_paths, path2tab='tab', path2trained='bestmodel', trials=5, batch_size=32)

print('finish')
