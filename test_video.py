from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from PIL import Image
from IPython.display import display 
import matplotlib.pyplot as plt
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision.models as models
import torchsummary
import numpy as np
from collections import namedtuple

import re
import cv2

from test_config import *


with torch.no_grad():
    style_model = TransformerNet()

    ckpt_model_path = os.path.join(checkpoint_dir, checkpoint_file)
    checkpoint = torch.load(ckpt_model_path, map_location=device)

    # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
    for k in list(checkpoint.keys()):
        if re.search(r'in\d+\.running_(mean|var)$', k):
            del checkpoint[k]

    style_model.load_state_dict(checkpoint['model_state_dict'])
    style_model.to(device)

    cap = cv2.VideoCapture(source_file) #FIXME

    frame_cnt = 0
    
    fourcc = cv2.VideoWriter_fourcc(*'X264') 
    out = cv2.VideoWriter(output_file, fourcc, 60.0, (1280,688)) #FIXME       
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        
        try:
            frame = frame[:,:,::-1] - np.zeros_like(frame)
        except:
            break

        content_image = frame
        content_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
        content_image = content_transform(content_image)
        content_image = content_image.unsqueeze(0).to(device)

        output = style_model(content_image).cpu()
        save_image(debug_folder + str(frame_cnt) +".png", output[0]) #FIXME
#             out.write(post_process_image(output[0]))
        frame_cnt += 1
        

    cap.release()
    out.release()
    cv2.destroyAllWindows()

