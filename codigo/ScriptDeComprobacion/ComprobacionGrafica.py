# %%
import tensorflow as tf

# %%
print("¿GPU detectada?:", tf.config.list_physical_devices('GPU'))


import torch
print(torch.version.cuda)
