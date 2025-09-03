#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   metauas.py
@Time    :   2025/03/26 23:46:12
@Author  :   Bin-Bin Gao
@Email   :   csgaobb@gmail.com
@Homepage:   https://csgaobb.github.io/
@Version :   1.0
@Desc    :   some classes and functions for MetaUAS
'''


import os
import random
import kornia as K
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import tqdm
import time
import cv2

from PIL import Image
from einops import rearrange
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms.functional import pil_to_tensor
from segmentation_models_pytorch.unet.model import UnetDecoder
from segmentation_models_pytorch.fpn.decoder import FPNDecoder
from segmentation_models_pytorch.encoders import get_encoder, get_preprocessing_params

def set_random_seed(seed=233, reproduce=False):
    np.random.seed(seed)
    torch.manual_seed(seed ** 2)
    torch.cuda.manual_seed(seed ** 3)
    random.seed(seed ** 4)

    if reproduce:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True 

def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=np.float32)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)


def read_image_as_tensor(path_to_image):
    pil_image = Image.open(path_to_image).convert("RGB")
    image_as_tensor = pil_to_tensor(pil_image).float() / 255.0
    return image_as_tensor

def safely_load_state_dict(model, checkpoint):
    model.load_state_dict(torch.load(checkpoint), strict=True)
    return model

def visualizer(paths, anomaly_map, img_size, save_path, cls_name, shot=0):
    for idx, path in enumerate(paths):
        cls = path.split('/')[-2]
        filename = path.split('/')[-1]
        vis = cv2.cvtColor(cv2.resize(cv2.imread(path), (img_size, img_size)), cv2.COLOR_BGR2RGB)  # RGB
        #mask = normalize(anomaly_map[idx])
        mask = anomaly_map[idx]
        vis = apply_ad_scoremap(vis, mask)
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)  # BGR
        save_vis = os.path.join(save_path, 'imgs-shot' + str(shot), cls_name[idx], cls)
        if not os.path.exists(save_vis):
            os.makedirs(save_vis)
        cv2.imwrite(os.path.join(save_vis, filename), vis)

def few_shot(memory, query):
    simscore = torch.einsum("bcij,bckl->bijkl", F.normalize(query, dim=1),  F.normalize(memory, dim=1))
    simscore = rearrange(simscore, "b h1 w1 h2 w2 -> b h1 w1 (h2 w2)")

    M = 1/2 * torch.min(1.0 - simscore, dim = -1)[0]
    return M

class AlignmentModule(nn.Module):
    def __init__(self, input_channels=2048, hidden_channels=256, alignment_type="sa", fusion_policy='cat'):
        super().__init__()
        self.fusion_policy = fusion_policy
        self.alignment_layer = AlignmentLayer(input_channels, hidden_channels, alignment_type=alignment_type)
    
    def forward(self, query_features, prompt_features):
            if isinstance(prompt_features, list):
                aligned_prompt = []
                for i in range(len(prompt_features)):
                    weighted_prompt.append(self.alignment_layer(query_features, prompt_features[i]))
                aligned_prompt = torch.mean(torch.stack(aligned_prompt),0)
                
            else:
                aligned_prompt = self.alignment_layer(query_features, prompt_features)
           
            if self.fusion_policy == 'cat':
                query_features = rearrange(
                    [query_features, aligned_prompt], "two b c h w -> b (two c) h w"
                )
            elif self.fusion_policy == 'add':
                query_features = query_features + aligned_prompt
                
            elif self.fusion_policy == 'absdiff':
                query_features = (query_features - aligned_prompt).abs()
        
            return query_features
        
class AlignmentLayer(nn.Module):
    def __init__(self, input_channels=2048, hidden_channels=256, alignment_type="sa"):
        super().__init__()
        self.alignment_type = alignment_type
        if alignment_type != "na":
            self.dimensionality_reduction = nn.Conv2d(
                input_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=True
            )

    def forward(self, query_features, prompt_features):
        # no-alignment
        if self.alignment_type == 'na':
            return prompt_features
        else:
            Q = self.dimensionality_reduction(query_features)
            K = self.dimensionality_reduction(prompt_features)
            V = rearrange(prompt_features, "b c h w -> b c (h w)")

            soft_attention_map = torch.einsum("bcij,bckl->bijkl", Q, K)
            soft_attention_map = rearrange(soft_attention_map, "b h1 w1 h2 w2 -> b h1 w1 (h2 w2)")
            soft_attention_map = nn.Softmax(dim=3)(soft_attention_map)

            # soft-alignment
            if self.alignment_type == 'sa':
                aligned_features = torch.einsum("bijp,bcp->bcij", soft_attention_map, V)
            # hard-alignment
            if self.alignment_type == 'ha':
                max_v, max_index = attention_map.max(dim=-1, keepdim=True)
                hard_attention_map = (attention_map == max_v).float()
                aligned_features = torch.einsum("bijp,bcp->bcij", hard_attention_map, V)

            return aligned_features


class MetaUAS(pl.LightningModule):
    def __init__(self, encoder_name, decoder_name, encoder_depth, decoder_depth, num_alignment_layers, alignment_type, fusion_policy):
        super().__init__()
        
        self.encoder_name = encoder_name
        self.decoder_name = decoder_name 
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth

        self.num_alignment_layers = num_alignment_layers
        self.alignment_type = alignment_type
        self.fusion_policy = fusion_policy
        
      
        align_input_channels = [448, 160, 56]
        align_hidden_channels = [224, 80, 28]
        encoder_channels = [3, 48, 32, 56, 160, 448] 
        decoder_channels = [256, 128, 64, 64, 48]
        
        self.encoder = get_encoder(
            self.encoder_name,
            in_channels=3,
            depth=self.encoder_depth,
            weights="imagenet",)

        preparams = get_preprocessing_params(
            self.encoder_name, 
            pretrained="imagenet"
            )
        
        self.preprocess = transforms.Normalize(preparams['mean'], preparams['std'])

        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        if self.decoder_name == "unet":
            encoder_out_channels = encoder_channels[self.encoder_depth-self.decoder_depth:]
            if self.fusion_policy == 'cat':
                num_alignment_layers = self.num_alignment_layers
            elif self.fusion_policy == 'add' or self.fusion_policy == 'absdiff':
                num_alignment_layers = 0

            self.decoder = UnetDecoder(
                encoder_channels=encoder_out_channels,
                decoder_channels=decoder_channels,
                n_blocks= self.decoder_depth,
                attention_type="scse",
                num_coam_layers= num_alignment_layers,
            )
             
        elif self.decoder_name == "fpn":
            encoder_out_channels = encoder_channels
            if self.fusion_policy == 'cat':
                for i in range(self.num_alignment_layers):
                    encoder_out_channels[-(i+1)] = 2 * encoder_out_channels[-(i+1)]
            
            self.decoder = FPNDecoder(
                encoder_channels= encoder_out_channels,
                encoder_depth=self.encoder_depth,
                pyramid_channels=256,
                segmentation_channels=decoder_channels[-1],
                dropout=0.2,
                merge_policy="add",
            )

        elif self.decoder_name == "fpnadd":
            segmentation_channels = 256 #128
            encoder_out_channels = encoder_channels
            if self.fusion_policy == 'cat':
                for i in range(self.num_alignment_layers):
                    encoder_out_channels[-(i+1)] = 2 * encoder_out_channels[-(i+1)]
            
            self.decoder = FPNDecoder(
                encoder_channels= encoder_out_channels,
                encoder_depth=self.encoder_depth,
                pyramid_channels=256,
                segmentation_channels=segmentation_channels,
                dropout=0.2,
                merge_policy="add",
            )
        elif self.decoder_name == "fpncat":
            encoder_out_channels = encoder_channels
            segmentation_channels = 256 #128
            if self.fusion_policy == 'cat':
                for i in range(self.num_alignment_layers):
                    encoder_out_channels[-(i+1)] = 2 * encoder_out_channels[-(i+1)]
            
            self.decoder = FPNDecoder(
                encoder_channels= encoder_out_channels,
                encoder_depth=self.encoder_depth,
                pyramid_channels=256,
                segmentation_channels=segmentation_channels,
                dropout=0.2,
                merge_policy="cat",
            )
             
         
        if self.alignment_type == "sa" or self.alignment_type == "na" or  self.alignment_type == "ha" :
            self.alignment = nn.ModuleList(
                [
                    AlignmentModule(
                        input_channels=align_input_channels[i],
                        hidden_channels=align_hidden_channels[i],
                        alignment_type=self.alignment_type,
                        fusion_policy=self.fusion_policy,
                    )
                    for i in range(self.num_alignment_layers)
                ]
            )
       
        if self.decoder_name == "fpncat":
            self.mask_head = nn.Conv2d(
                segmentation_channels*4,
                1,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        elif self.decoder_name == "fpnadd":
            self.mask_head = nn.Conv2d(
                segmentation_channels,
                1,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        else:
            self.mask_head = nn.Conv2d(
                decoder_channels[-1],
                1,
                kernel_size=1,
                stride=1,
                padding=0,
            )
           
    def forward(self, batch):
        query_input = self.preprocess(batch["query_image"])
        prompt_input = self.preprocess(batch["prompt_image"]) 

        with torch.no_grad():
            query_encoded_features = self.encoder(query_input)
            prompt_encoded_features = self.encoder(prompt_input)
                
        for i in range(len(self.alignment)):
            query_encoded_features[-(i + 1)] = self.alignment[i](query_encoded_features[-(i + 1)], prompt_encoded_features[-(i + 1)])
        
        query_decoded_features = self.decoder(*query_encoded_features[self.encoder_depth-self.decoder_depth:])
        
        if self.decoder_name == "fpn" or self.decoder_name == "fpncat" or self.decoder_name == "fpnadd":
            output = F.interpolate(self.mask_head(query_decoded_features), scale_factor=4, mode="bilinear", align_corners=False) 
            
        elif self.decoder_name == "unet":
            if self.decoder_depth == 4:
                output = F.interpolate(self.mask_head(query_decoded_features), scale_factor=2, mode="bilinear", align_corners=False) 
            if self.decoder_depth == 5:
                if not self.training:
                    output = self.mask_head(query_decoded_features)

        return output.sigmoid()
