#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   demo_metauas.py
@Time    :   2025/03/26 23:49:14
@Author  :   Bin-Bin Gao
@Email   :   csgaobb@gmail.com
@Homepage:   https://csgaobb.github.io/
@Version :   1.0
@Desc    :   MetaUAS Demo
'''


import os
import cv2
import torch
import json
import shutil
import kornia as K
import numpy as np

from easydict import EasyDict
from argparse import ArgumentParser
from metauas import MetaUAS, set_random_seed, normalize, apply_ad_scoremap, read_image_as_tensor, safely_load_state_dict

if __name__ == "__main__":
    random_seed = 1
    
    set_random_seed(random_seed)

    ckt_path = 'weights/metauas-256.ckpt'
    img_size = 256
    #ckt_path = "weights/metauas-512.ckpt"
    #img_size = 512

    # load model
    encoder = 'efficientnet-b4' 
    decoder = 'unet' 
    encoder_depth = 5
    decoder_depth = 5
    num_crossfa_layers = 3
    alignment_type =  'sa' 
    fusion_policy = 'cat'

    model = MetaUAS(encoder, 
                decoder, 
                encoder_depth, 
                decoder_depth, 
                num_crossfa_layers, 
                alignment_type, 
                fusion_policy
                )       


    model = safely_load_state_dict(model, ckt_path)
    model.cuda()
    model.eval()
    

    # load test images
    path_root = "./images/"
    path_to_prompt = path_root + "036.png" 
    path_to_query = path_root + "024.png"

    query = read_image_as_tensor(path_to_query)
    prompt = read_image_as_tensor(path_to_prompt)

    if query.shape[1] != img_size:
        resize_trans = K.augmentation.Resize([img_size, img_size], return_transform=True)
        query = resize_trans(query)[0]
        prompt = resize_trans(prompt)[0]
    

    test_data = {
            "query_image": query.cuda(),
            "prompt_image": prompt.cuda(),
        }
     
    # forward
    predicted_masks = model(test_data)

    # visualization
    query_img = test_data["query_image"][0] * 255
    query_img = query_img.permute(1,2,0)

    pred = (1-predicted_masks.squeeze().detach())[:, :, None].cpu().numpy().repeat(3, 2)
    # normalize just for analysis
    scoremap_self = apply_ad_scoremap(query_img.cpu(), normalize(pred))
    cv2.imwrite('./anomaly_map.jpg', scoremap_self)
   
