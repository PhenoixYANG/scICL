import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.manifold import TSNE
import umap
from matplotlib import pyplot as plt
from src.models import ZINB,log_zinb_positive
import os
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from tqdm import tqdm
def ZINB_AE_train(AE,AEtype,device,train_loader,optimizer,epoch):
    epoch_loss = 0.0
    total_samples = 0
    for batch_idx, (rna,atac,rna_s,atac_s,rna_raw,atac_raw,y) in tqdm(enumerate(train_loader)):
        rna,atac,rna_s,atac_s,rna_raw,atac_raw,y=rna.to(device),atac.to(device),rna_s.to(device),atac_s.to(device),rna_raw.to(device),atac_raw.to(device),y.to(device)
        optimizer.zero_grad()
        if AEtype=='rna':
            o_rna,pi_rna, disp_rna, mean_rna, z_rna,=AE(rna,rna_s)
            loss=ZINB(pi_rna,disp_rna,rna,o_rna,ridge_lambda=1.0)
        if AEtype=='atac':    
            o_atac,pi_atac, disp_atac, mean_atac, z_atac=AE(atac,atac_s)
            loss=ZINB(pi_atac,disp_atac,atac,o_atac,ridge_lambda=1.0)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * y.size(0)
        total_samples += y.size(0)
    average_loss = epoch_loss / total_samples
    return average_loss
        
            
def contrastive_train(model,device,train_loader,optimizer,epoch):
    model.train()
    epoch_loss = 0.0
    epoch_loss1=0.0
    epoch_loss2=0.0
    epoch_loss_mse1=0.0
    epoch_loss_mse2=0.0
    epoch_loss_itc=0.0
    epoch_loss_itm=0.0

    total_samples = 0
    for batch_idx, (rna,atac,rna_s,atac_s,rna_raw,atac_raw,y) in tqdm(enumerate(train_loader)):
        rna,atac,rna_s,atac_s,rna_raw,atac_raw,y=rna.to(device),atac.to(device),rna_s.to(device),atac_s.to(device),rna_raw.to(device),atac_raw.to(device),y.to(device)
        optimizer.zero_grad()
        z_loss1,z_loss2,mse_loss1,mse_loss2,itc_loss,itm_loss,z=model(rna,atac,rna_s,atac_s,rna_raw,atac_raw,y)
        loss=model.config['x1']*z_loss1+model.config['x2']*z_loss2+model.config['x3']*itc_loss+model.config['x4']*itm_loss+mse_loss1+mse_loss2
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * y.size(0)
        epoch_loss1 +=z_loss1.item() * y.size(0)
        epoch_loss2+= z_loss2.item() * y.size(0)
        epoch_loss_itc+=itc_loss.item() * y.size(0)
        epoch_loss_itm+=itm_loss.item() * y.size(0)
        epoch_loss_mse1+=mse_loss1.item() * y.size(0)
        epoch_loss_mse2+=mse_loss2.item() * y.size(0)
        total_samples += y.size(0)
    average_loss = epoch_loss / total_samples
    loss1=epoch_loss1 / total_samples
    loss2=epoch_loss2 / total_samples
    loss_itc=epoch_loss_itc / total_samples
    loss_itm=epoch_loss_itm / total_samples
    loss_mse1=epoch_loss_mse1 / total_samples
    loss_mse2=epoch_loss_mse2/ total_samples
    return average_loss,loss1,loss2,loss_itc,loss_itm,loss_mse1,loss_mse2


def contrastive_test(model,device,test_loader,epoch=0):
    model.eval()
    test_loss = 0
    emb=[]
    label=[]
    with torch.no_grad():
        for batch_idx, (rna,atac,rna_s,atac_s,rna_raw,atac_raw,y) in enumerate(test_loader):
            rna,atac,rna_s,atac_s,rna_raw,atac_raw,y=rna.to(device),atac.to(device),rna_s.to(device),atac_s.to(device),rna_raw.to(device),atac_raw.to(device),y.to(device)
            z_loss1,z_loss2,mse_loss1,mse_loss2,itc_loss,itm_loss,z=model(rna,atac,rna_s,atac_s,rna_raw,atac_raw,y)
            loss=z_loss1+z_loss2+itc_loss+itm_loss+mse_loss1+mse_loss2
            emb.append(z.cpu().numpy())
            label.append(y.cpu().numpy())
    emb=np.concatenate(emb, axis=0)
    labels=np.concatenate(label, axis=0)
    log_dir=model.config['log_dir']
    reduce=umap.UMAP(n_neighbors=100, n_components=2, metric="euclidean")
    z_umap = reduce.fit_transform(emb)
    kmeans = KMeans(n_clusters=model.config['n_clusters'],n_init=10,random_state=3407)
    pred = kmeans.fit_predict(z_umap)
    cm = confusion_matrix(pred, labels)
    row_ind, col_ind = linear_sum_assignment(-cm)
    label_mapping = {col_ind[i]: row_ind[i] for i in range(len(col_ind))}
    pred = np.array([label_mapping[label] for label in pred])
    plt.scatter(z_umap[:, 0], z_umap[:, 1], c=labels)
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.savefig(os.path.join(log_dir,f'contrastive_learning_{epoch}.png'))
    plt.close()
    np.save(os.path.join(log_dir,'label.npy'), labels)
    ari = adjusted_rand_score(labels, pred)
    nmi = normalized_mutual_info_score(labels, pred)
    print(f'ari:{ari}\n')
    print(f'nmi:{nmi}\n')
    with open(os.path.join(log_dir,'result.txt'),'a') as fw:
        fw.write(f'ari:{ari}\n')
        fw.write(f'nmi:{nmi}\n\n')
    return ari,nmi,z_umap,emb
