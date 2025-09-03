'''
 # @ Author: Bin-Bin Gao
 # @ Create Time: 2025-09-02 16:08:23
 # @ Modified by: Bin-Bin Gao
 # @ Modified time: 2025-09-03 22:15:52
 # @ Description: test script for MetaUAS
 '''


import os 
import shutil
import cv2
import json
import random
import numpy as np
from tqdm import tqdm 

from argparse import ArgumentParser

import kornia as K
import matplotlib.patches as patches
import matplotlib.pyplot as plt

import shapely.geometry
from datetime import datetime
from tabulate import tabulate

import torch
import torch.nn.functional as F
import torchmetrics


from torch.utils.data import DataLoader
from dataset import ADDataset
from metauas import MetaUAS, visualizer, set_random_seed, normalize, apply_ad_scoremap, safely_load_state_dict
#from eval_metric.torch_metric import Evaluator
cpu_eva = False
from eval_metric.sklearn_metric import Evaluator
cpu_eva = True 


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="") 
    parser.add_argument("--seed", type=int, default="1") 
    parser.add_argument("--image_dir", type=str, default="") 
    parser.add_argument("--img_size", type=int, default="256") 
    parser.add_argument("--test_json", type=str, default="") 
    parser.add_argument("--prompt_json", type=str, default="") 
    parser.add_argument("--eval_metrics", type=str, nargs="+", default=['I-AUROC', 'I-AP', 'I-F1max', 'P-AUROC', 'P-AP', 'P-F1max', 'P-AUPRO'], help='evaluation metrics')   
    parser.add_argument("--save_path", type=str, default='./temp') 
    args = parser.parse_args()        

    random_seed = args.seed
    set_random_seed(random_seed)
    img_size = args.img_size
    ckt_path = args.checkpoint

    # init model
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

    # load pre-trained model
    model = safely_load_state_dict(model, ckt_path)
    model.cuda()
    model.eval()

    device = torch.device("cuda")
    method = 'metauas'
    image_size = img_size
    
    # load testing dataset
    dataset = ADDataset(args.image_dir, args.test_json, args.img_size, prompt_meta_file=args.prompt_json)

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
    )
    
    # one-shot testing
    if cpu_eva:
        evaluator = Evaluator('cpu', metrics=args.eval_metrics)
    else:
        evaluator = Evaluator(device, metrics=args.eval_metrics)

    gt_masks, pr_masks, cls_names, gt_anomalys, pr_anomalys, img_paths = [], [], [], [], [], []
    for batch_data in tqdm(dataloader):
        test_data = {
            "query_image": batch_data["query_image"].cuda(),
            "prompt_image": batch_data["prompt_image"].cuda()

        }

        cls_name = batch_data['cls_name']
        gt_label = batch_data['query_label'].cuda()
        gt_mask = batch_data['query_mask'].cuda()
        gt_mask[gt_mask > 0] = 1
        query_path = batch_data['query_filename']

        with torch.no_grad():
            pr_mask = model(test_data)
           
        cls_names.append(np.array(cls_name))
        img_paths.append(np.array(query_path))

        if cpu_eva:
            gt_masks.append(gt_mask.cpu())
            gt_anomalys.append(gt_label.cpu())
            pr_masks.append(pr_mask.cpu())  
        else:
            gt_masks.append(gt_mask)
            gt_anomalys.append(gt_label)
            pr_masks.append(pr_mask)

        # save visualization results
        visualizer(query_path, pr_mask[:,0].detach().cpu().numpy(), args.img_size, args.save_path, cls_name)

        
            
    nowtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    results_eval = dict(
                        gt_masks=gt_masks, 
                        pr_masks=pr_masks, 
                        cls_names=cls_names, 
                        gt_anomalys=gt_anomalys, 
                        img_paths=img_paths
                        )
    results_eval = {k: np.concatenate(v, axis=0) if k in ['cls_names', 'img_paths'] else torch.cat(v, dim=0) for k, v in results_eval.items()}
    
   
    # print results
    obj_list = list(sorted(set(results_eval['cls_names'])))
    msg = {}
    for idx, cls_name in enumerate(tqdm(obj_list)): 
        metric_results = evaluator.run(results_eval, cls_name)
        msg['Name'] = msg.get('Name', [])
        msg['Name'].append(cls_name)
        avg_act = True if len(obj_list) > 1 and idx == len(obj_list) - 1 else False
        msg['Name'].append('Avg') if avg_act else None
    
        for metric in args.eval_metrics:
            metric_result = metric_results[metric] #* 100

            msg[metric] = msg.get(metric, [])
            msg[metric].append(metric_result)
            
            if avg_act:
                metric_result_avg = sum(msg[metric]) / len(msg[metric])
                msg[metric].append(metric_result_avg)
                
    tab = tabulate(msg, headers='keys', tablefmt="pipe", floatfmt='.3f', numalign="center", stralign="center", ) 
    print(tab)