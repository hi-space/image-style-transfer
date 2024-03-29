import copy
import time
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import datasets
import torchsummary
import os
from PIL import Image

from train_config import *
from models import *
from utils import *


if __name__ == "__main__":
    device = get_device()

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    style_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
    ])

    train_dataset = datasets.ImageFolder("coco", transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

    transformer = TransformerNet().to(device)
    vgg = VGG16(requires_grad=False).to(device)

    optimizer = torch.optim.Adam(transformer.parameters(), initial_lr)
    mse_loss = nn.MSELoss()

    style = load_image(filename=style_image_location, size=None, scale=None)
    style = style_transform(style)
    style = style.repeat(batch_size, 1, 1, 1).to(device)

    features_style = vgg(normalize_batch(style))
    gram_style = [gram_matrix(y) for y in features_style]

    if transfer_learning:
        checkpoint = torch.load(ckpt_model_path, map_location=device)
        transformer.load_state_dict(checkpoint['model_state_dict'])
        transformer.to(device)

        transfer_learning_epoch = checkpoint['epoch'] 
    else:
        transfer_learning_epoch = 0

    for epoch in range(transfer_learning_epoch, num_epochs):
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0

        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()

            x = x.to(device)
            y = transformer(x)

            y = normalize_batch(y)
            x = normalize_batch(x)

            features_y = vgg(y.to(device))
            features_x = vgg(x.to(device))

            content_loss = content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
            style_loss *= style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()

            if (batch_id + 1) % log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), epoch + 1, count, len(train_dataset),
                                    agg_content_loss / (batch_id + 1),
                                    agg_style_loss / (batch_id + 1),
                                    (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                print(mesg)

            if checkpoint_dir is not None and (batch_id + 1) % checkpoint_interval == 0:
                transformer.eval().cpu()
                ckpt_model_filename = "ckpt_epoch_" + str(epoch) + "_batch_id_" + str(batch_id + 1) + ".pth"
                print(str(epoch), "th checkpoint is saved!")
                ckpt_model_path = os.path.join(checkpoint_dir, ckpt_model_filename)
                torch.save({
                'epoch': epoch,
                'model_state_dict': transformer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss
                }, ckpt_model_path)

                transformer.to(device).train()  