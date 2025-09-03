'''
 # @ Author: Bin-Bin Gao
 # @ Create Time: 2025-09-02 22:39:56
 # @ Modified by: Bin-Bin Gao
 # @ Modified time: 2025-09-02 22:40:23
 # @ Description: some classes and functions for MetaUAS
 '''

import glob
import logging
import os
import numpy as np
import tabulate
import torch
import torch.nn.functional as F
from sklearn import metrics
from prettytable import PrettyTable
from skimage import measure
import time

def cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3):
    # ref: https://github.com/gudovskiy/cflow-ad/blob/master/train.py
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                pro.append(tp_pixels / region.area)
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        pros.append(np.array(pro).mean())
        fprs.append(fpr)
        ths.append(th)
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]


    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    pro_auc = metrics.auc(fprs, pros[idxes])
    return pro_auc

class Evaluator(object):
    def __init__(self, device, metrics=[]):
        if len(metrics) == 0:
            self.metrics = [
                'I-AUROC', 'I-AP', 'I-F1max', 
                'P-AUROC', 'P-AP', 'P-F1max', 'P-AUPRO'
            ]
        else:
            self.metrics = metrics

    def run(self, results, cls_name):
        idxes = results['cls_names'] == cls_name
        gt_px = results['gt_masks'][idxes] 
        pr_px = results['pr_masks'][idxes]

        gt_sp = results['gt_anomalys'][idxes]
        #pr_sp = results['pr_anomalys'][idxes]
        
        pr_px = (pr_px - pr_px.min())/(pr_px.max() - pr_px.min())
        pr_sp = pr_px.max(dim=-1)[0].max(dim=-1)[0]
        
        if len(gt_px.shape) == 4:
            gt_px = gt_px.squeeze(1)
        if len(pr_px.shape) == 4:
            pr_px = pr_px.squeeze(1)

        pr_px = pr_px.numpy() 
        gt_px = gt_px.numpy()

        pr_sp = pr_sp.numpy()
        gt_sp = gt_sp.numpy()

    
        eval_results = {}
        for metric in self.metrics:
            if metric.startswith('I-AUROC'):
                fpr, tpr, thresholds = metrics.roc_curve(gt_sp.ravel(), pr_sp.ravel(), pos_label=1)
                i_auc = metrics.auc(fpr, tpr)
                if i_auc < 0.5:
                    i_auc = 1 - i_auc
                eval_results[metric] = i_auc
                
            elif metric.startswith('P-AUROC'):
                fpr, tpr, thresholds = metrics.roc_curve(gt_px.ravel(), pr_px.ravel(), pos_label=1)
                p_auc = metrics.auc(fpr, tpr)
                if p_auc < 0.5:
                    p_auc = 1 - p_auc
                eval_results[metric] = p_auc
                    
            elif metric.startswith('I-AP'):
                eval_results[metric] = metrics.average_precision_score(gt_sp.ravel(), pr_sp.ravel(), pos_label=1, average=None)
                
            elif metric.startswith('P-AP'):
                eval_results[metric] = metrics.average_precision_score(gt_px.ravel(), pr_px.ravel(), pos_label=1, average=None)
                    
            elif metric.startswith('I-F1max'):
                precisions, recalls, thresholds = metrics.precision_recall_curve(gt_sp.ravel(), pr_sp.ravel())
                f1_scores = (2 * precisions * recalls) / (precisions + recalls)
                eval_results[metric] = np.max(f1_scores[np.isfinite(f1_scores)])
            
            elif metric.startswith('P-F1max'):
                precisions, recalls, thresholds = metrics.precision_recall_curve(gt_px.ravel(), pr_px.ravel())
                f1_scores = (2 * precisions * recalls) / (precisions + recalls)
                eval_results[metric] = np.max(f1_scores[np.isfinite(f1_scores)])
            
            elif metric.startswith('P-AUPRO'):
                eval_results[metric] = cal_pro_score(gt_px, pr_px)
            
        return eval_results