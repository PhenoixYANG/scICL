from PIL import Image
import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader,Sampler
from torch import optim, nn, utils, Tensor
import torch.nn.functional as F
from sklearn.cluster import KMeans
import umap
from matplotlib import pyplot as plt
from info_nce import InfoNCE, info_nce
import random
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

class MeanAct(nn.Module):
    def forward(self, x):
        return torch.clamp(torch.exp(x), 1e-5, 1e6)
    
class DispAct(nn.Module):
    def forward(self, x):
        return torch.clamp(x, 1e-4, 1e4)

def _nan2zero(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x), x)

def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + float('inf'), x)
def _nelem(x):
    nelem = torch.sum(~torch.isnan(x).float())
    return torch.where(nelem == 0., torch.tensor(1.), nelem).to(x.dtype)

class ZINB_AE_att(nn.Module):
    def __init__(self,in_dim, hidden1, hidden2, hidden3, z_emb_size, dropout_rate):
        super(ZINB_AE_att,self).__init__()
        self.in_dim = in_dim
        self.emb_size=z_emb_size
        self.fc_encoder = nn.Sequential(
            nn.Linear(self.in_dim, hidden1),
            nn.Sigmoid(),
            # nn.Dropout(p=dropout_rate),

            nn.Linear(hidden1, hidden2),
            nn.Sigmoid(),
            # nn.Dropout(p=dropout_rate),

            nn.Linear(hidden2, hidden3),
            nn.Sigmoid(),
            # nn.Dropout(p=dropout_rate),

            nn.Linear(hidden3, z_emb_size),
            nn.Sigmoid(),
            # nn.Dropout(p=dropout_rate),
        )

        # Distribution reconstruction
        self.fc_decoder = nn.Sequential(
            nn.Linear(z_emb_size, hidden3),
            nn.Sigmoid(),

            nn.Linear(hidden3, hidden2),
            nn.Sigmoid(),

            nn.Linear(hidden2, hidden1),
            nn.Sigmoid(),
            nn.Linear(hidden1, self.in_dim),
            nn.Sigmoid(),
        )
        self.pi = nn.Sequential(
            nn.Linear(self.in_dim,self.in_dim),
            nn.Sigmoid()
        )
        self.disp = nn.Sequential(
            nn.Linear(self.in_dim,self.in_dim),
            DispAct()
        )
        self.mean = nn.Sequential(
            nn.Linear(self.in_dim,self.in_dim),
            MeanAct()
        )





    def encode(self, x):
        emb = self.fc_encoder(x)
        return emb
    def decode(self,emb,scale_factor=1.0):
        latent = self.fc_decoder(emb)
        pi = torch.sigmoid(self.pi(latent))
        disp = self.disp(latent)
        mean = self.mean(latent)
        output = mean * scale_factor.unsqueeze(1).expand(mean.shape)
        return output, pi, disp, mean

    def forward(self, x,scale_factor=1.0):
        emb = self.fc_encoder(x)
        latent = self.fc_decoder(emb)
        pi = torch.sigmoid(self.pi(latent))
        disp = self.disp(latent)
        mean = self.mean(latent)
        output = mean * scale_factor.unsqueeze(1).expand(mean.shape)
        return output, pi, disp, mean, emb


class pretrain_ZINB_AE_att(nn.Module):
    def __init__(self,config,d):
        super(pretrain_ZINB_AE_att,self).__init__()
        self.config=config
        self.training_step_outputs = []
        self.AE=ZINB_AE(d,self.config['hidden1'],self.config['hidden2'],self.config['hidden3'],self.config['emb_size'],self.config['dropout'])
        # self.AE.add_attention(self.config['n_head'])
        self.emb=[]
        self.label=[]
        self.mul_attention = nn.MultiheadAttention(self.config['emb_size'], self.config['n_head'],batch_first=True)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.ffn=nn.Sequential(
            nn.Linear(self.config['emb_size'],self.config['emb_size']*4),
            nn.ReLU(),
            nn.Linear(self.config['emb_size']*4,self.config['emb_size'])
        )




    def attention(self,xs):
        # x=torch.concat(xs,dim=1)
        x=xs
        att_out, att_weights = self.mul_attention(x,x,x)
        att_out=self.pool(att_out.permute(0,2,1)).squeeze()
        return att_out
    
    def encode(self,d):
        ds=self.generate_mask_view(d,self.config['mask_p'],self.config['n_view'])
        ds=torch.concat(ds,dim=1)
        with torch.no_grad():
            emb_ds=self.AE.encode(ds)
        att_emb=self.attention(emb_ds)
        att_emb=emb_ds[:, 0, :].squeeze()+att_emb
        return att_emb

    def forward(self,d,d_s):
        ds=self.generate_mask_view(d,self.config['mask_p'],self.config['n_view'])
        ds=torch.concat(ds,dim=1)
        emb_ds=self.AE.encode(ds)
        emb_d=self.AE.encode(d)
        att_emb=self.attention(emb_ds)
        att_emb=self.ffn(att_emb)
        o,pi, disp, mean=self.AE.decode(emb_d,d_s)
        return o,pi, disp, mean, att_emb,emb_d

    def generate_mask_view(self,x,mask_p,n_view):
        d=x.shape[1]
        masked_x=[]
        for i in range(n_view):
            column_mask = torch.rand(d) < mask_p  # 长度为 d 的布尔掩码
            n_x=x.clone()
            n_x[:, column_mask] = 0 
            masked_x.append(n_x.unsqueeze(1))
        return masked_x

    def save_AE(self,dataset,type):
        AE=self.AE.cpu()
        torch.save(AE.state_dict(),os.path.join(self.config['model_dir'],f'pretrain_{type}AE_{dataset}.pth'))




class ZINB_AE(nn.Module):
    def __init__(self,in_dim, hidden1, hidden2, hidden3, z_emb_size, dropout_rate):
        super(ZINB_AE,self).__init__()
        self.in_dim = in_dim
        self.fc_encoder = nn.Sequential(
            nn.Linear(self.in_dim, hidden1),
            nn.ReLU(),
            # nn.Dropout(p=dropout_rate),

            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            # nn.Dropout(p=dropout_rate),

            nn.Linear(hidden2, hidden3),
            nn.ReLU(),
            # nn.Dropout(p=dropout_rate),

            nn.Linear(hidden3, z_emb_size),
            nn.ReLU(),
            # nn.Dropout(p=dropout_rate),
        )

        # Distribution reconstruction
        self.fc_decoder = nn.Sequential(
            nn.Linear(z_emb_size, hidden3),
            nn.ReLU(),

            nn.Linear(hidden3, hidden2),
            nn.ReLU(),

            nn.Linear(hidden2, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, self.in_dim),
            nn.ReLU(),
        )
        self.pi = nn.Sequential(
            nn.Linear(self.in_dim,self.in_dim),
            nn.Sigmoid()
        )
        self.disp = nn.Sequential(
            nn.Linear(self.in_dim,self.in_dim),
            DispAct()
        )
        self.mean = nn.Sequential(
            nn.Linear(self.in_dim,self.in_dim),
            MeanAct()
        )


    def encode(self, x):
        emb = self.fc_encoder(x)
        return emb
    def decode(self,emb,scale_factor=1.0):
        latent = self.fc_decoder(emb)
        pi = torch.sigmoid(self.pi(latent))
        disp = self.disp(latent)
        mean = self.mean(latent)
        output = mean * scale_factor.unsqueeze(1).expand(mean.shape)
        return output, pi, disp, mean


    def forward(self, x,scale_factor=1.0):
        emb = self.fc_encoder(x)
        # expression matrix decoder
        latent = self.fc_decoder(emb)
        pi = torch.sigmoid(self.pi(latent))
        disp = self.disp(latent)
        mean = self.mean(latent)
        output = mean * scale_factor.unsqueeze(1).expand(mean.shape)
        return output, pi, disp, mean, emb


class Attention(nn.Module):
    def __init__(self, dropout):
        super(Attention, self).__init__()
        self.dropout = nn.Dropout(dropout)
    def forward(self,q,k,v):
        d=torch.tensor(q.shape[1]).to(q.device)
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(d)
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, v)
        return output, attention_weights



class ITCloss(nn.Module):
    def __init__(self, t):
        super(ITCloss, self).__init__()
        self.t = t
    def forward(self,rna,atac):
        rna = F.normalize(rna, dim=1)
        atac = F.normalize(atac, dim=1)
        logits = torch.matmul(rna, atac.T) / self.t
        labels = torch.arange(logits.size(0)).long().to(rna.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        return (loss_i2t + loss_t2i) / 2


def NB(theta, y_true, y_pred, mask=False, debug=False, mean=False):
    eps = 1e-8
    scale_factor = 1.0
    y_true = y_true.float()
    y_true=torch.max(y_true, torch.tensor(1e-4))
    y_pred = y_pred.float() * scale_factor
    if mask:
        nelem = _nelem(y_true)
        y_true = _nan2zero(y_true)
    theta = torch.min(theta, torch.tensor(1e6))
    t1 = torch.lgamma(theta + eps) + torch.lgamma(y_true + 1.0) - torch.lgamma(y_true + theta + eps)
    t2 = (theta + y_true) * torch.log(1.0 + (y_pred / (theta + eps))) + (y_true * (torch.log(theta + eps) - torch.log(y_pred + eps)))
    final = t1 + t2
    final = _nan2inf(final)
    if mean:
        if mask:
            final = torch.sum(final) / nelem
        else:
            final = torch.mean(final)
    return final


def log_zinb_positive(x, mu, theta, pi, eps=1e-8):
    if theta.ndimension() == 1:

        theta = theta.view(1, theta.size(0))

    softplus_pi = F.softplus(-pi)

    log_theta_eps = torch.log( theta + eps )

    log_theta_mu_eps = torch.log( theta + mu + eps )

    pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

    case_zero = F.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

    case_non_zero = (-softplus_pi + pi_theta_log
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1))

    mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)

    res = mul_case_zero + mul_case_non_zero
    result = - torch.sum(res, dim=1)
    result = _nan2inf(result)

    return result

def ZINB(pi, theta, y_true, y_pred, ridge_lambda, mean=True, mask=False, debug=False):
    eps = 1e-8
    scale_factor = 1.0
    y_pred=torch.clamp(y_pred, min=0)
    nb_case = NB(theta, y_true, y_pred, mean=False, debug=debug) - torch.log(1.0 - pi + eps)
    y_true = y_true.float()
    y_pred = y_pred.float() * scale_factor
    theta = torch.min(theta, torch.tensor(1e6))
    zero_nb = torch.pow(theta / (theta + y_pred + eps), theta)
    zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
    result = torch.where(y_true < 1e-8, zero_case, nb_case)
    ridge = ridge_lambda * torch.square(pi)
    result += ridge

    if mean:
        if mask:
            result = torch.mean(result)
        else:
            result = torch.mean(result)

    result = _nan2inf(result)
    return result




class pretrain_ZINB_AE(nn.Module):
    def __init__(self,config,d):
        super(pretrain_ZINB_AE,self).__init__()
        self.config=config
        self.training_step_outputs = []
        self.AE=ZINB_AE(d,self.config['hidden1'],self.config['hidden2'],self.config['hidden3'],self.config['emb_size'],self.config['dropout'])
        self.emb=[]
        self.label=[]

    def forward(self,d,d_s):
        o,pi, disp, mean, z=self.AE(d,d_s)
        return o,pi, disp, mean, z


    def save_AE(self,dataset,type):
        AE=self.AE.cpu()
        torch.save(AE.state_dict(),os.path.join(self.config['model_dir'],f'pretrain_{type}AE_{dataset}.pth'))




class contrastive_learning(nn.Module):
    def __init__(self,config,d_rna,d_atac):
        super(contrastive_learning,self).__init__()
        self.config=config
        self.norm_rna=nn.LayerNorm(normalized_shape=self.config['emb_size'])
        self.norm_atac=nn.LayerNorm(normalized_shape=self.config['emb_size'])  
        self.r2a_attention=Attention(config['attention_dropout'])
        self.a2r_attention=Attention(config['attention_dropout'])

        self.fc1=nn.Sequential(
            nn.Linear(2*self.config['emb_size'],self.config['emb_size']),
            nn.ReLU(),
            nn.Linear(self.config['emb_size'],self.config['emb_size']),
            nn.ReLU()
        )
        self.fc2=nn.Linear(self.config['emb_size'],2)
        self.rnaAE=pretrain_ZINB_AE_att(self.config,d_rna)
        self.atacAE=pretrain_ZINB_AE_att(self.config,d_atac)
        self.softmax=nn.Softmax(dim=1)
        self.itc=ITCloss(self.config['temperature'])
        self.mse_loss=nn.MSELoss()

        self.bce_loss=nn.BCEWithLogitsLoss()


    def forward(self,rna,atac,rna_s,atac_s,rna_raw,atac_raw,label):
        o_rna,pi_rna, disp_rna, mean_rna, att_rna,z_rna=self.rnaAE(rna,rna_s)
        o_atac,pi_atac, disp_atac, mean_atac,att_atac,z_atac=self.atacAE(atac,atac_s)
        mse_loss1=self.mse_loss(att_rna,z_rna)
        mse_loss2=self.mse_loss(att_atac,z_atac)
        z1_rna,_=self.r2a_attention(z_rna,z_atac,z_atac)
        z1_atac,_=self.a2r_attention(z_atac,z_rna,z_rna)
        z_rna=z_rna+z1_rna
        z_atac=z_atac+z1_atac
        z_rna=self.norm_rna(z_rna)
        z_atac=self.norm_atac(z_atac)

        z_loss1=ZINB(pi_rna,disp_rna,rna_raw,o_rna,ridge_lambda=1.0)
        z_loss2=ZINB(pi_atac,disp_atac,atac_raw,o_atac,ridge_lambda=1.0)
        itc_loss=self.itc(z_rna,z_atac)
        new_rna,new_atac,new_label=self.negative_construct(z_rna,z_atac,label)
        new_label=new_label.to(label.device)
        p_z=torch.concat([new_rna,new_atac],dim=1)
        p_z=self.fc1(p_z)
        pred=self.softmax(self.fc2(p_z))
        itm_loss=F.cross_entropy(pred, new_label) 
        with torch.no_grad():
            z=torch.concat([z_rna,z_atac],dim=1)
            z=self.fc1(z)
            pred=self.softmax(self.fc2(z))

        return z_loss1,z_loss2,mse_loss1,mse_loss2,itc_loss,itm_loss,z


    def negative_construct(self,rna,atac,label):
        new_rna=[]
        new_atac=[]
        new_label=[]
        for index in range(label.shape[0]):
            target_label=label[index].item()
            negative_index = [i for i in range(label.shape[0]) if label[i].item() != target_label and i != index]
            num_negative_samples = min(self.config['n_sample'], len(negative_index))
            samples = np.random.choice(negative_index, size=num_negative_samples, replace=False)
            new_rna+=[rna[index].unsqueeze(0)]*(num_negative_samples+1)
            new_atac+=[atac[index].unsqueeze(0)]
            new_atac+=[atac[i].unsqueeze(0) for i in samples]
            new_label+=[torch.tensor(1).unsqueeze(0)]
            new_label+=[torch.tensor(0).unsqueeze(0)]*num_negative_samples
        new_rna=torch.cat(new_rna,dim=0)
        new_atac=torch.cat(new_atac,dim=0)
        new_label=torch.cat(new_label,dim=0)
        return new_rna,new_atac,new_label
        
    def loadAE(self,dataset):
        self.rnaAE.AE.load_state_dict(torch.load(os.path.join(self.config['model_dir'],f'pretrain_rnaAE_{dataset}.pth')))
        self.atacAE.AE.load_state_dict(torch.load(os.path.join(self.config['model_dir'],f'pretrain_atacAE_{dataset}.pth')))