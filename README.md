# [**NeurIPS 24**] MetaUAS

[![HuggingFace Space](https://img.shields.io/badge/🤗-HuggingFace%20Space-cyan.svg)](https://huggingface.co/spaces/csgaobb/MetaUAS)

> Official PyTorch Implementation of [MetaUAS: Universal Anomaly Segmentation with One-Prompt Meta-Learning](https://proceedings.neurips.cc/paper_files/paper/2024/file/463a91da3c832bd28912cd0d1b8d9974-Paper-Conference.pdf), NeurIPS 2024.



## Introduction 
MetaUAS unifies anomaly segmentation into change segmentation and provides a pure visual foundation model, which requires only one normal image prompt and no additional training, and effectively and efficiently segments any visual anomalies.  MetaUAS significantly outperforms most zero-shot, few-shot, and even full-shot anomaly segmentation methods. 

## MetaUAS Framework
![overview](images/MetaUAS_Framework.jpg)


## Main Results

![main-results](images/MetaUAS-Main-Results.jpg)

![main-com-eff](images/MetaUAS-Com-Eff.jpg)

![main-diffprompts](images/MetaUASwithDiffprompts.jpg)

![main-diffprompts](images/MetaUAS-Vis.jpg)

## Demo
You can run the demo application using the following command:
```
python app.py
```
You could also try to [Online Demo](https://huggingface.co/spaces/csgaobb/MetaUAS) for a quick start. 
Note that our online demo is based on CPU since there is no GPU for Hugging Face Space.




## Citing
If you find this code useful in your research, please consider citing us:

```
@inproceedings{gao2024metauas,
  title  = {MetaUAS: Universal Anomaly Segmentation with One-Prompt Meta-Learning},
  author = {Gao, Bin-Bin},
  booktitle = {Advances in Neural Information Processing Systems},
  pages = {39812--39836},
  year = {2024}
}
```
