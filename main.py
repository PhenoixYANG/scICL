from src.train import *
from src.get_config import get_config
from src.data_process import *
from src.models import *
from src.train import *
from src.utils import *
import json
import os
import numpy
import torch
import argparse


from torch.utils.data import DataLoader,Sampler

def main():
    parser = argparse.ArgumentParser(description='Example script to demonstrate argparse.')
    parser.add_argument('-d','--device', type=str, help='device num',default='2')
    parser.add_argument('-c','--config', type=str, help='config type',default='pbmc_10k')
    parser.add_argument('-m','--module', type=str, help='train module',default='pretrain')
    args = parser.parse_args()
    seed = 3407

    os.environ['CUDA_VISIBLE_DEVICES']=args.device
    torch.manual_seed(seed)  # CPU 相关操作设置随机种子
    torch.cuda.manual_seed(seed)  # GPU 相关操作设置随机种子
    np.random.seed(seed)
    config_dir=f'/remote-home/share/dmb_nas/liuwuchao/cell/train/config/{args.config}.json'
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config=get_config(config_dir)
    # torch.autograd.set_detect_anomaly(True)
    if model_config['dataset']=='pbmc_10k' or model_config['dataset']=='ma_2020' or model_config['dataset']=='chen_2019':
        train_data,test_data=load_pbmc_dataset(model_config["dataset"],model_config["rna_dir"],model_config["atac_dir"],model_config)
    elif model_config['dataset']=='cellmix':
        train_data,test_data=load_cellmix_dataset(model_config["rna_dir"],model_config["atac_dir"],model_config["label"],model_config)
    elif model_config['dataset']=='pbmc_3k':
        train_data,test_data=load_pbmc3k_dataset(model_config["data_dir"],model_config)
    if args.module=='pretrain':
        # '''
        p_config=model_config['pretrainAE']
        log_dir = generate_log(os.path.join('logs','pretrain_AE',model_config['dataset']))
        p_config['log_dir']=log_dir
        train_loader=DataLoader(train_data,batch_size=p_config['batch_size'],shuffle=False,num_workers=p_config['num_workers'])
        test_loader=DataLoader(test_data,batch_size=p_config['batch_size'],shuffle=False,num_workers=p_config['num_workers'])
        N1,N2=train_data.get_dim()
        p_config['n_clusters']=train_data.get_n_clusters()
        rnaAE=pretrain_ZINB_AE(p_config,N1).to(device)
        atacAE=pretrain_ZINB_AE(p_config,N2).to(device)
        rna_optimizer = optim.Adam(rnaAE.parameters(), lr=p_config['lr'])
        atac_optimizer = optim.Adam(atacAE.parameters(), lr=p_config['lr'])
        fw=open(os.path.join(log_dir,'AE_loss.txt'),'w')
        for i in range(p_config['epoches']):
            print(f'epoch: {i}')
            rnaloss=ZINB_AE_train(rnaAE,'rna',device,train_loader,rna_optimizer,i)
            atacloss=ZINB_AE_train(atacAE,'atac',device,train_loader,atac_optimizer,i)
            fw.write(f'epoch {i}: rnaloss={rnaloss} atacloss={atacloss}\n')
        fw.close()
        rnaAE.save_AE(model_config['dataset'],'rna')
        atacAE.save_AE(model_config['dataset'],'atac')
        with open(os.path.join(log_dir,'config.json'),'w') as fw:
            json.dump(model_config,fw,indent=4)
    elif args.module=='contrastive_learning':
        c_config=model_config['contrastive_learning']
        log_dir = generate_log(os.path.join('logs','contrastive_learning',model_config['dataset']))
        c_config['log_dir']=log_dir
        train_loader=DataLoader(train_data,batch_size=c_config['batch_size'],shuffle=False,num_workers=c_config['num_workers'])
        test_loader=DataLoader(test_data,batch_size=c_config['batch_size'],shuffle=False,num_workers=c_config['num_workers'])
        N1,N2=train_data.get_dim()
        c_config['n_clusters']=train_data.get_n_clusters()
        CL=contrastive_learning(c_config,N1,N2)
        CL.loadAE(model_config['dataset'])
        CL=CL.to(device)
        optimizer = optim.Adam(CL.parameters(), lr=c_config['lr'])
        fw=open(os.path.join(log_dir,'contrastive_loss.txt'),'w')
        best=0
        for i in range(c_config['epoches']):
            print(f'epoch: {i}')
            loss,loss1,loss2,loss_itc,loss_itm,loss_mse1,loss_mse2=contrastive_train(CL,device,train_loader,optimizer,i)
            fw.write(f'epoch {i}: loss={loss} loss1={loss1} loss2={loss2} loss_itc={loss_itc} loss_itm={loss_itm} loss_mse1={loss_mse1} loss_mse2={loss_mse2}\n')
        fw.close()
        contrastive_test(CL,device,train_loader,c_config['epoches'])
        with open(os.path.join(log_dir,'config.json'),'w') as fw:
            json.dump(model_config,fw,indent=4)
        
    



if __name__=='__main__':
    main()