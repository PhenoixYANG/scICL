# scICL: Integrating Single-cell Multi-omics by Intra- and Inter-omics Contrastive Learning for Clustering Analysis
Code for our proposed scICL, which is a deep single-cell multi-omics data integration framework utilizing intra- and inter-omics contrastive learning for cell clustering analysis.
![scICL](/figs/scICL.png)

## Installation

To install the necessary packages, use conda and the provided environment.yaml file:

`conda env create -f environment.yaml`

## Running

To reproduce experimental results of scICL on PBMC-10K dataset, you can run 

`mkdir log`

`mkdir log/test`

`python test_with_checkpoint.py -d '0' --config 'your_path/config/pbmc_10k.json' --checkpoint path_to_checkpoint`

The PBMC-10k-dataset can be downloaded  from [here](https://scglue.readthedocs.io/zh-cn/latest/data.html). Please replace the data path in 'config/pbmc_10k.json'  with the path of the dataset you downloaded

The pretrained checkpoint can be downloaded from [here](https://pan.quark.cn/s/52bfb960f8fe)

To train a scICL model, you can run

'python main.py -d '0' --config your_path_forconfig --module 'pretrain''
'python main.py -d '0' --config your_path_forconfig  --module 'contrastive_learning '

The PBMC-3K dataset can be downloaded from [here](https://github.com/xianglin226/scMDC/tree/master/datasets).

The Ma-2020 dataset can be downloaded from [here](https://scglue.readthedocs.io/zh-cn/latest/data.html).

The scRNA-seq and scATAC-seq count matrices of CellMix dataset can be downloaded from [here](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE126074). And the cell type infomation can be found at `data/cellmix_cell_metadata.txt`.

## Baselines

The implementation of baseline methods in our paper can be found as follow:

**scVI** [https://docs.scvi-tools.org/en/stable/tutorials/notebooks/scrna/harmonization.html](https://docs.scvi-tools.org/en/stable/tutorials/notebooks/scrna/harmonization.html)

**scGPT** [https://github.com/bowang-lab/scGPT](https://github.com/bowang-lab/scGPT)

**PeakVI** [https://docs.scvi-tools.org/en/stable/tutorials/notebooks/atac/PeakVI.html](https://docs.scvi-tools.org/en/stable/tutorials/notebooks/atac/PeakVI.html)

**scMVAE** [https://github.com/cmzuo11/scMVAE](https://github.com/cmzuo11/scMVAE)

**VIMCCA** [https://scbean.readthedocs.io/en/latest/tutorials/index.html#tutorials-for-scbean-vimcca](https://scbean.readthedocs.io/en/latest/tutorials/index.html#tutorials-for-scbean-vimcca)

**SCMCs** [https://www.sdu-idea.cn/codes.php?name=scMCs](https://www.sdu-idea.cn/codes.php?name=scMCs)

**DCCA** [https://github.com/cmzuo11/DCCA](https://github.com/cmzuo11/DCCA) 

**scMCL** [https://github.com/yuxuanChen777/scMLC](https://github.com/yuxuanChen777/scMLC)
