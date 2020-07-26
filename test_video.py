import cv2
import os
import re
import torch
import torchvision.transforms as transforms
import torchsummary
import numpy as np

from PIL import Image

from test_config import *
from models import *
from utils import get_device, save_image


if __name__ == "__main__":
    device = get_device()

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
        
        fourcc = cv2.VideoWriter_fourcc(*'mpeg') 
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
            save_image(debug_dir + str(frame_cnt) +".png", output[0]) 
    #             out.write(post_process_image(output[0]))
            frame_cnt += 1

        cap.release()
        out.release()
        cv2.destroyAllWindows()

