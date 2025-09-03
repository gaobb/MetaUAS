'''
 # @ Author: Bin-Bin Gao
 # @ Create Time: 2025-09-03 19:27:40
 # @ Modified by: Bin-Bin Gao
 # @ Modified time: 2025-09-03 22:15:17
 # @ Description: some classes and functions for MetaUAS
 '''


import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
import utils.geometry

class ADDataset(Dataset):
    def __init__(self, image_dir, test_meta_file, image_size, random_seed=1, prompt_meta_file=None):
        self.image_dir = image_dir
        self.split = 'test'
        self.random_seed = random_seed
        self.img_size = image_size
   
        with open(test_meta_file, "r") as f_r:
            self.metas = []
            for line in f_r:
                meta = json.loads(line)
                #if meta["clsname"] == 'pill':
                self.metas.append(meta)

        self.prompts = {}
        if 'oneprompt_seed' in prompt_meta_file:
            self.optional_prompt = False
            with open(prompt_meta_file, 'r') as f:
                self.prompts = json.load(f)
            
        else:
            self.optional_prompt = True
            with open(prompt_meta_file, 'r') as f:
                for line in f:
                    meta = json.loads(line)
                    self.prompts.update(meta)
 
    def __len__(self):
        return len(self.metas)
    
    def read_image_as_tensor(self, path_to_image):
        pil_image = Image.open(path_to_image).convert("RGB")
        image_as_tensor = pil_to_tensor(pil_image).float() / 255.0
        return image_as_tensor
    
    def read_mask_as_tensor(self, path_to_image):
        pil_mask = Image.open(path_to_image).convert("L")
        tensor_mask = pil_to_tensor(pil_mask)>0
        mask_as_tensor =  tensor_mask.float()

        return mask_as_tensor
   
    
    def __getitem__(self, item_index):
        item_data = self.__base_getitem__(item_index)

        return item_data 

    def __base_getitem__(self, item_index):
        image_meta = self.metas[item_index]
        label = image_meta["label"]
        image_filename = image_meta["filename"]
        path_to_image = os.path.join(self.image_dir, image_meta["filename"])
    
        if self.optional_prompt:
            prompt_filename = self.prompts[image_filename][0]
            path_to_prompt  = os.path.join(self.image_dir, prompt_filename)
        else:
            prompt_meta = self.prompts[image_meta["clsname"]]
            prompt_filename = prompt_meta["filename"]
            path_to_prompt  = os.path.join(self.image_dir, prompt_meta["filename"])
        
        image = self.read_image_as_tensor(path_to_image)
        prompt = self.read_image_as_tensor(path_to_prompt)

         # read / generate mask
        if image_meta.get("maskname", None):
            path_to_mask = os.path.join(self.image_dir, image_meta["maskname"])
            mask1  = self.read_mask_as_tensor(path_to_mask)
        else:
            if label == 0:  # good
                mask1 = np.zeros((1, image.shape[1], image.shape[2]))
            elif label == 1:  # defective
                mask1 = np.ones((1, image.shape[1], image.shape[2]))
            else:
                raise ValueError("Labels must be [None, 0, 1]!")
        mask1 = torch.as_tensor(mask1)
        
        # resize
        (
            image,
            prompt,
            mask1,
        ) = utils.geometry.resize_pairimages_and_mask(
            image,
            prompt,
            (self.img_size, self.img_size),
            mask1,
        )
        mask2 = mask1

        return {
            "query_image": image,
            "prompt_image": prompt,
            "cls_name": image_meta["clsname"],
            "query_filename": os.path.join(self.image_dir, image_filename),
            "prompt_filename": prompt_filename,
            "query_label": label,
            "query_mask": mask1,
            "prompt_mask": mask2,
        }