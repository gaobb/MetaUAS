#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   app.py
@Time    :   2025/03/26 23:48:24
@Author  :   Bin-Bin Gao
@Email   :   csgaobb@gmail.com
@Homepage:   https://csgaobb.github.io/
@Version :   1.0
@Desc    :   MetaUAS Demo with Gradio
'''


import os
import cv2
import torch
import json
import shutil
import kornia as K
import numpy as np
import gradio as gr
from easydict import EasyDict
from argparse import ArgumentParser
from torchvision.transforms.functional import pil_to_tensor

from metauas import MetaUAS, set_random_seed, normalize, apply_ad_scoremap, safely_load_state_dict

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# configurations
random_seed = 1
encoder_name = 'efficientnet-b4' 
decoder_name = 'unet' 
encoder_depth = 5
decoder_depth = 5
num_alignment_layers = 3
alignment_type =  'sa' 
fusion_policy = 'cat'


# build model
set_random_seed(random_seed)
metauas_model = MetaUAS(encoder_name, 
                decoder_name, 
                encoder_depth, 
                decoder_depth, 
                num_alignment_layers, 
                alignment_type, 
                fusion_policy
                )   

def process_image(prompt_img, query_img, options):
    # Load the model based on selected options
    if 'model-512' in options:
        ckt_path = "weights/metauas-512.ckpt"
        model = safely_load_state_dict(metauas_model, ckt_path)
        img_size = 512
    else:
        ckt_path = 'weights/metauas-256.ckpt'
        model = safely_load_state_dict(metauas_model, ckt_path)
        img_size = 256

    model.to(device)
    model.eval()

    # Ensure image is in RGB mode
    prompt_img = prompt_img.convert('RGB')
    query_img = query_img.convert('RGB')

    query_img = pil_to_tensor(query_img).float() / 255.0
    prompt_img = pil_to_tensor(prompt_img).float() / 255.0
    
    if query_img.shape[1] != img_size:
        resize_trans = K.augmentation.Resize([img_size, img_size], return_transform=True)
        query_img = resize_trans(query_img)[0]
        prompt_img = resize_trans(prompt_img)[0]

    
    test_data = {
            "query_image": query_img.to(device),
            "prompt_image": prompt_img.to(device),
        }

    
    # Forward
    with torch.no_grad():
        predicted_masks = model(test_data) 
        anomaly_score = predicted_masks[:].max()

    # Process anomaly map
    query_img = test_data["query_image"][0] * 255
    query_img = query_img.permute(1,2,0)

    anomaly_map = predicted_masks.squeeze().detach()[:, :, None].cpu().numpy().repeat(3, 2)
    
    anomaly_map_vis = apply_ad_scoremap(query_img.cpu(), normalize(anomaly_map))


    anomaly_map = (anomaly_map * 255).astype(np.uint8)
    anomaly_map = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)
    anomaly_map = cv2.cvtColor(anomaly_map, cv2.COLOR_BGR2RGB)
    
    return anomaly_map_vis, anomaly_map, f'{anomaly_score:.3f}'

# Define examples
examples = [
    ["images/134.png", "images/000.png", "model-256"],
    ["images/036.png", "images/024.png", "model-256"],
    ["images/178.png", "images/003.png", "model-256"],
]

# Gradio interface layout
with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center" style='margin-top: 30px;'>MetaUAS: Universal Anomaly Segmentation</h1>""")
    gr.HTML("""<h1 align="center" style="font-size: 15px; "style='margin-top: 40px;'>just given ONE normal image prompt</h1>""")
    
    with gr.Row():
        with gr.Column():
            with gr.Row():
                prompt_image = gr.Image(type="pil", label="Prompt Image")
                query_image = gr.Image(type="pil", label="Query Image")
            model_selector = gr.Radio(["model-256", "model-512"], label="Pre-models")
        
        with gr.Column():
            with gr.Row():
                 anomaly_map_vis = gr.Image(type="pil", label="Anomaly Results")
                 anomaly_map = gr.Image(type="pil", label="Anomaly Maps")
            anomaly_score = gr.Textbox(label="Anomaly Score")

    with gr.Row():
        submit_button = gr.Button("Submit", elem_id="submit-button")
        clear_button = gr.Button("Clear")

    # Set up the event handlers
    submit_button.click(process_image, inputs=[prompt_image, query_image, model_selector], outputs=[anomaly_map_vis, anomaly_map, anomaly_score])
    clear_button.click(lambda: (None, None, None), outputs=[anomaly_map_vis, anomaly_map, anomaly_score])

    # Add examples directly to the Blocks interface
    gr.Examples(examples, inputs=[prompt_image, query_image, model_selector])

# Add custom CSS to control the output image size and button styles
demo.css = """
#submit-button {
    color: red !important;  /* Font color */
    background-color: orange !important;  /* Background color */
    border: none !important;  /* Remove border */
    padding: 10px 20px !important;  /* Add padding */
    border-radius: 5px !important;  /* Rounded corners */
    font-size: 16px !important;  /* Font size */
    cursor: pointer !important;  /* Pointer cursor on hover */
}

#submit-button:hover {
    background-color: darkorange !important;  /* Darker orange on hover */
}
"""

# Launch the demo
demo.launch()

