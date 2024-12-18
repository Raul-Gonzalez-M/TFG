# %%
import tensorflow as tf

# %%
print("¿GPU detectada?:", tf.config.list_physical_devices('GPU'))
strategy = tf.distribute.MirroredStrategy()
print(f"Número de GPUs detectadas: {strategy.num_replicas_in_sync}")

import torch
print(torch.version.cuda)
