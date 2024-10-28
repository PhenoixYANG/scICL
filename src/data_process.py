import anndata as ad
import networkx as nx
import scanpy as sc
import pandas as pd
import numpy as np
from matplotlib import rcParams
from pybedtools import BedTool
import mygene
import scglue
from scipy import stats, spatial, sparse
import torch
import os
import h5py 
from torch.utils.data.dataset import Dataset
from sklearn.model_selection import train_test_split

def geneSelection(data, threshold=0, atleast=10, 
                  yoffset=.02, xoffset=5, decay=1.5, n=None, 
                  plot=True, markers=None, genes=None, figsize=(6,3.5),
                  markeroffsets=None, labelsize=10, alpha=1, verbose=1):
    if sparse.issparse(data):
        zeroRate = 1 - np.squeeze(np.array((data>threshold).mean(axis=0)))
        A = data.multiply(data>threshold)
        A.data = np.log2(A.data)
        meanExpr = np.zeros_like(zeroRate) * np.nan
        detected = zeroRate < 1
        meanExpr[detected] = np.squeeze(np.array(A[:,detected].mean(axis=0))) / (1-zeroRate[detected])
    else:
        zeroRate = 1 - np.mean(data>threshold, axis=0)
        meanExpr = np.zeros_like(zeroRate) * np.nan
        detected = zeroRate < 1
        mask = data[:,detected]>threshold
        logs = np.zeros_like(data[:,detected]) * np.nan
        logs[mask] = np.log2(data[:,detected][mask])
        meanExpr[detected] = np.nanmean(logs, axis=0)


    lowDetection = np.array(np.sum(data>threshold, axis=0)).squeeze() < atleast
    zeroRate[lowDetection] = np.nan
    meanExpr[lowDetection] = np.nan
            
    if n is not None:
        up = 10
        low = 0
        for t in range(100):
            nonan = ~np.isnan(zeroRate)
            selected = np.zeros_like(zeroRate).astype(bool)
            selected[nonan] = zeroRate[nonan] > np.exp(-decay*(meanExpr[nonan] - xoffset)) + yoffset
            if np.sum(selected) == n:
                break
            elif np.sum(selected) < n:
                up = xoffset
                xoffset = (xoffset + low)/2
            else:
                low = xoffset
                xoffset = (xoffset + up)/2
        if verbose>0:
            print('Chosen offset: {:.2f}'.format(xoffset))
    else:
        nonan = ~np.isnan(zeroRate)
        selected = np.zeros_like(zeroRate).astype(bool)
        selected[nonan] = zeroRate[nonan] > np.exp(-decay*(meanExpr[nonan] - xoffset)) + yoffset
    if plot:
        if figsize is not None:
            plt.figure(figsize=figsize)
        plt.ylim([0, 1])
        if threshold>0:
            plt.xlim([np.log2(threshold), np.ceil(np.nanmax(meanExpr))])
        else:
            plt.xlim([0, np.ceil(np.nanmax(meanExpr))])
        x = np.arange(plt.xlim()[0], plt.xlim()[1]+.1,.1)
        y = np.exp(-decay*(x - xoffset)) + yoffset
        if decay==1:
            plt.text(.4, 0.2, '{} genes selected\ny = exp(-x+{:.2f})+{:.2f}'.format(np.sum(selected),xoffset, yoffset), 
                     color='k', fontsize=labelsize, transform=plt.gca().transAxes)
        else:
            plt.text(.4, 0.2, '{} genes selected\ny = exp(-{:.1f}*(x-{:.2f}))+{:.2f}'.format(np.sum(selected),decay,xoffset, yoffset), 
                     color='k', fontsize=labelsize, transform=plt.gca().transAxes)

        plt.plot(x, y, color=sns.color_palette()[1], linewidth=2)
        xy = np.concatenate((np.concatenate((x[:,None],y[:,None]),axis=1), np.array([[plt.xlim()[1], 1]])))
        t = plt.matplotlib.patches.Polygon(xy, color=sns.color_palette()[1], alpha=.4)
        plt.gca().add_patch(t)
        
        plt.scatter(meanExpr, zeroRate, s=1, alpha=alpha, rasterized=True)
        if threshold==0:
            plt.xlabel('Mean log2 nonzero expression')
            plt.ylabel('Frequency of zero expression')
        else:
            plt.xlabel('Mean log2 nonzero expression')
            plt.ylabel('Frequency of near-zero expression')
        plt.tight_layout()
        
        if markers is not None and genes is not None:
            if markeroffsets is None:
                markeroffsets = [(0, 0) for g in markers]
            for num,g in enumerate(markers):
                i = np.where(genes==g)[0]
                plt.scatter(meanExpr[i], zeroRate[i], s=10, color='k')
                dx, dy = markeroffsets[num]
                plt.text(meanExpr[i]+dx+.1, zeroRate[i]+dy, g, color='k', fontsize=labelsize)
        plt.savefig('data.png')
    return selected

def normalize(adata, filter_min_counts=True, size_factors=True, normalize_input=False, logtrans_input=True):
    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)
    eps=1e-8
    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if logtrans_input:
        sc.pp.log1p(adata)

    if size_factors:
        adata.obs['size_factors'] = np.log(np.clip(np.sum(adata.X, axis=1), 1, None)+eps)
    else:
        adata.obs['size_factors'] = 1.0

    if normalize_input:
        sc.pp.scale(adata)

    return adata

def normalize_atac(adata, filter_min_counts=True, size_factors=True, normalize_input=False, logtrans_input=True, binary=True):
    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)
    eps=1e-8
    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if logtrans_input:
        sc.pp.log1p(adata)

    if size_factors:
        adata.obs['size_factors'] = np.log(np.clip(np.sum(adata.X, axis=1), 1, None)+eps)
    else:
        adata.obs['size_factors'] = 1.0
    if normalize_input:
        sc.pp.scale(adata)
    if binary:
        adata.X=np.clip(adata.X.toarray(),0,1)
    return adata



def process_data(rna,atac,config):

    if config['filter']:
        print('Filting data----------------------')
        sc.pp.highly_variable_genes(rna, n_top_genes=config['n_rna'], flavor='seurat_v3')
        rna = rna[:, rna.var['highly_variable']]
        sc.pp.highly_variable_genes(atac, n_top_genes=config['n_atac'], flavor='seurat_v3')
        atac = atac[:, atac.var['highly_variable']]
    rna=normalize(rna, size_factors=True, normalize_input=True, logtrans_input=True)
    atac=normalize(atac, size_factors=True, normalize_input=True, logtrans_input=True)
    atac_valid_cells = set(atac.obs_names)
    rna_valid_cells = set(rna.obs_names)
    common_cells = atac_valid_cells & rna_valid_cells
    common_cells=list(common_cells)
    nlabel=[]
    atac = atac[common_cells, :]
    rna = rna[common_cells, :]
    rna_raw,classes,label=extract_data(rna)
    atac_raw,classes,label=extract_data(atac)
    return rna,atac,classes,label


def extract_data(data):
    cell_type=data.obs['cell_type'].tolist()
    classes, label_ground_truth=np.unique(cell_type, return_inverse=True) #得到细胞类型标签
    matrix=data.X
    matrix = matrix.toarray() if not isinstance(matrix, np.ndarray) else matrix #得到归一化细胞技术矩阵
    return matrix,classes,label_ground_truth
    

def split_data(x1,x2,y,test_size=0.1):
    x1_train,x1_test,x2_train,x2_test,y_train,y_test= train_test_split(x1, x2, y, test_size=test_size, random_state=42)
    return x1_train,x1_test,x2_train,x2_test,y_train,y_test


class scdata(Dataset):
    def __init__(self,rna,atac,label,classes):
        super(scdata,self).__init__()
        self.rna_x=torch.tensor(rna.X.toarray() if not isinstance(rna.X, np.ndarray) else rna.X)
        self.rna_s=torch.tensor(rna.obs.size_factors)
        self.rna_raw=torch.tensor(rna.raw.X.toarray() if not isinstance(rna.raw.X, np.ndarray) else rna.raw.X)
        
        self.atac_x=torch.tensor(atac.X.toarray() if not isinstance(atac.X, np.ndarray) else atac.X)
        self.atac_s=torch.tensor(atac.obs.size_factors)
        self.atac_raw=torch.tensor(atac.raw.X.toarray() if not isinstance(atac.raw.X, np.ndarray) else atac.raw.X)

        self.label=torch.tensor(label)
        self.classes=classes
        print(self.rna_x.shape)
        print(self.atac_x.shape)
        print(self.label.shape)

    

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.rna_x[index],self.atac_x[index],self.rna_s[index],self.atac_s[index],self.rna_raw[index],self.atac_raw[index],self.label[index]


    def get_dim(self):
        return self.rna_x.shape[1],self.atac_x.shape[1]

    def get_n_clusters(self):
        return len(self.classes)

def get_data(rna,atac,config):
    rna_x,atac_x,classes,label=process_data(rna,atac,config)
    print('Generating dataset----------------------')
    train_data=scdata(rna_x,atac_x,label,classes)
    test_data=train_data
    return train_data

def load_cellmix_dataset(rna,atac,label,config):
    x1  = sc.read(rna).transpose().X
    x2=sc.read(atac).transpose().X
    info=pd.read_table(label, header=0, index_col=0 )
    classes, label=np.unique(info['cell_line'].values, return_inverse=True)
    if config['filter']:
        print('Filting data----------------------')
        importantGenes=geneSelection(x1, n=config['n_rna'], plot=False)
        x1 = x1[:, importantGenes]
        min_cells=int(x2.shape[0] * 0.05)
        sc.pp.filter_genes(x2, min_cells=min_cells)
    rna = sc.AnnData(x1)
    atac = sc.AnnData(x2)
    rna=normalize(rna, size_factors=True, normalize_input=True, logtrans_input=True)
    atac=normalize(atac, size_factors=True, normalize_input=True, logtrans_input=True)
    train_data=scdata(rna,atac,label,classes)
    test_data=train_data
    return train_data


def load_pbmc3k_dataset(data,config):
    data_mat = h5py.File(data)
    x1 = np.array(data_mat['X1'],dtype=np.float32)
    x2 = np.array(data_mat['X2'],dtype=np.float32)
    y = np.array(data_mat['Y'],dtype=np.float32)
    classes = np.unique(y)
    rna = sc.AnnData(x1)
    atac = sc.AnnData(x2)
    if config['filter']:
        print('Filting data----------------------')
        sc.pp.highly_variable_genes(rna, n_top_genes=config['n_rna'], flavor='seurat_v3')
        rna = rna[:, rna.var['highly_variable']]
        sc.pp.highly_variable_genes(atac, n_top_genes=config['n_atac'], flavor='seurat_v3')
        atac = atac[:, atac.var['highly_variable']]
    rna=normalize(rna, size_factors=True, normalize_input=True, logtrans_input=True)
    atac=normalize(atac, size_factors=True, normalize_input=True, logtrans_input=True)
    train_data=scdata(rna,atac,y,classes)
    test_data=train_data
    return train_data


def load_pbmc_dataset(dataset,rna_dir,atac_dir,config):
    print('Loading data----------------------')
    rna = ad.read_h5ad(rna_dir)
    atac=ad.read_h5ad(atac_dir)  
    return get_data(rna,atac,config)


