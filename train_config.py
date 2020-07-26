import os
from PIL import Image


style_image_location = "image/dali-the persistence of memory.jpg"

style_image_sample = Image.open(style_image_location, 'r')

batch_size = 8
random_seed = 10
num_epochs = 64 
initial_lr = 1e-3
checkpoint_dir = "dali/"

content_weight = 1e5
style_weight = 1e10
log_interval = 50
checkpoint_interval = 500

transfer_learning = False 
ckpt_model_path = os.path.join(checkpoint_dir, "ckpt_epoch_63_batch_id_500.pth") 
