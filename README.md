<div align="center">
  
  <div>
  <h1>Learning to Cooperate for Continual Learning with Pre-training and Parameter-Efficient Fine-Tuning</h1>
  </div>
  <div>
      Dongyang Zhang; Zixuan Lu; Jummin Liu
  </div>
  <br/>
</div>

[//]: # (![本地路径]&#40;./resources/overview.pdf&#41;)
<img src=".\resources\overview.pdf">  

## Introduction
This repository is the PyTorch Implementation for paper: 
[Learning to Cooperate for Continual Learning with Pre-training and Parameter-Efficient Fine-Tuning]().
Our code will be released soon.

## Requirements
- torch==1.13.1
- torchvision==0.14.1
- timm==0.6.7
- pillow==9.2.0
- matplotlib==3.5.3
- torchprofile==0.0.4
- urllib3==2.0.3
- scipy==1.7.3
- scikit-learn==1.0.2
- numpy==1.21.6
- tqdm

Run the following command to setup the environment.
```
conda create -n l2c python=3.8
conda activate l2c
pip3 install -r requirements.txt
```

## Features
### Datasets
- [x] CIFAR10/100 (https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)
- [x] ImageNet-R (https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar)
- [x] CUB200 (https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz)
- [x] StanfordCars196 ()

### Pre-Training Checkpoints
- [x] [ViT-B/16 IN21k] will be automatically downloaded and loaded.
- [x] [ViT-B/16 MoCo v3]()
- [x] [DeiT-S/16 IN611]()

### PEFT implementations (Working in Progress)
- [x] LoRA
- [ ] Prompt 
- [ ] Prefix
- [ ] Adapter


## Preparation for Datasets and Pre-training weights
Download the datasets and place them in a directory. In my case, I place them at ```/home/zdy/mydatasets```.
Then you have to modify the default value of ```data-root``` in the `parse_args` function in `main.py`.

Likewise, place the downloaded checkpoints in one specific directory and modify ```pretraining-root``` .

### Training and evaluation
Take ```CIFAR100``` with ```vitb_in21k``` as an example, you can run the below command to for training and evaluating with 4 gpus.

```
export devices=0,1,2,3
export dataset='cifar100'
export pretrain='vitb_in21k'

# Train expert selector
CUDA_VISIBLE_DEVICES=$devices python -m torch.distributed.launch \
      --nproc_per_node=4 \
      --master_port=22323 \
      --use_env main.py \
      --config ./configs/cl_lora_${dataset}_10t_${pretrain}.py

# Train experts (validate with only one selected expert)
CUDA_VISIBLE_DEVICES=$devices python -m torch.distributed.launch \
      --nproc_per_node=4 \
      --master_port=22325 \
      --use_env main.py \
      --config-tp ./configs/cl_lora_${dataset}_10t_${pretrain}.py \
      --config ./configs/l2c_lora_${dataset}_10t_${pretrain}.py

# validate with top-k aggregation
CUDA_VISIBLE_DEVICES=$devices python -m torch.distributed.launch \
      --nproc_per_node=4 \
      --master_port=22326 \
      --use_env main.py \
      --eval-only \
      --cfg-options topk_val=5\
      --config-tp ./configs/cl_lora_${dataset}_10t_${pretrain}.py \
      --config ./configs/l2c_lora_${dataset}_10t_${pretrain}.py
```
Note, when training experts, the evaluation is based on only **one single expert** to save evaluation time.
Therefore, you can ignore the evaluation results during training.

The final evaluation with **TopK experts** will be done after all experts and tasks are finished. 
And the model ckpts and training logs will be saved in the ```output/cl_lora_${dataset}_10t_${pretrain}``` and ```output/l2c_lora_${dataset}_10t_${pretrain}```directory.

## Acknowledgement
This repository is mainly based on [HiDe-Prompt](https://github.com/thu-ml/HiDe-Prompt) and [SLCA](https://github.com/GengDavid/SLCA). We extend our gratitude to the contributors of these projects.
```bibtex
@article{wang2023hide,
  title={Hierarchical Decomposition of Prompt-Based Continual Learning: Rethinking Obscured Sub-optimality},
  author={Wang, Liyuan and Xie, Jingyi and Zhang, Xingxing and Huang, Mingyi and Su, Hang and Zhu, Jun},
  journal={Advances in Neural Information Processing Systems},
  year={2023}
}

@misc{zhang2024slcaunleashpowersequential,
      title={SLCA++: Unleash the Power of Sequential Fine-tuning for Continual Learning with Pre-training}, 
      author={Zhang, Gengwei and Wang, Liyuan and Kang, Guoliang and Chen, Ling and Wei, Yunchao},
      year={2024},
      eprint={2408.08295},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2408.08295}, 
}
```
## Citation
If you find this repository useful, please cite using this BibTeX:
```
None
```