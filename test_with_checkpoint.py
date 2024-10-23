from src.train import *
from src.get_config import get_config
from src.data_process import *
from src.models import *
from src.train import *
from src.utils import *
import json
import os
import numpy as np
import torch
import argparse


from torch.utils.data import DataLoader,Sampler

def main():
    parser = argparse.ArgumentParser(description='Example script to demonstrate argparse.')
    parser.add_argument('-d','--device', type=str, help='device num',default='2')
    parser.add_argument('-c','--config', type=str, help='config type',default='pbmc_10k')
    parser.add_argument('-m','--module', type=str, help='train module',default='pretrain')
    parser.add_argument('-p','--checkpoint', type=str, help='checkpoint',default='')
    args = parser.parse_args()
    seed = 3407

    os.environ['CUDA_VISIBLE_DEVICES']=args.device
    torch.manual_seed(seed)  # CPU 相关操作设置随机种子
    torch.cuda.manual_seed(seed)  # GPU 相关操作设置随机种子
    np.random.seed(seed)
    config_dir=f'/remote-home/share/dmb_nas/liuwuchao/cell/train/config/{args.config}.json'
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config=get_config(config_dir)
    model_checkpoint=args.checkpoint
    # torch.autograd.set_detect_anomaly(True)
    if model_config['dataset']=='pbmc_10k' or model_config['dataset']=='ma_2020' or model_config['dataset']=='chen_2019':
        train_data,test_data=load_pbmc_dataset(model_config["dataset"],model_config["rna_dir"],model_config["atac_dir"],model_config)
    elif model_config['dataset']=='cellmix':
        train_data,test_data=load_cellmix_dataset(model_config["rna_dir"],model_config["atac_dir"],model_config["label"],model_config)
    elif model_config['dataset']=='pbmc_3k':
        train_data,test_data=load_pbmc3k_dataset(model_config["data_dir"],model_config)
    c_config=model_config['contrastive_learning']
    log_dir = generate_log(os.path.join('logs','test',model_config['dataset']))
    c_config['log_dir']=log_dir
    print(f'log save in {log_dir}')
    
    train_loader=DataLoader(train_data,batch_size=c_config['batch_size'],shuffle=False,num_workers=c_config['num_workers'])
    N1,N2=train_data.get_dim()
    c_config['n_clusters']=train_data.get_n_clusters()
    CL=torch.load(model_checkpoint)
    CL.config=c_config
    CL=CL.to(device)
    ari,nmi,umap,emb=contrastive_test(CL,device,train_loader,c_config['epoches'])
    np.save(os.path.join(log_dir,'umap.npy'), umap)
    np.save(os.path.join(log_dir,'emb.npy'), emb)
        
    






if __name__=='__main__':
    main()