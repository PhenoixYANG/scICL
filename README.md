#scICL: Integrating Single-cell Multi-omics by Intra- and Inter-omics Contrastive Learning for Clustering Analysis
Code for our proposed scICL, which is a deep single-cell multi-omics data integration framework utilizing intra- and inter-omics contrastive learning for cell clustering analysis.
![scICL](/pics/scICL.png)

## Installation

To install the necessary packages, use conda and the provided environment.yaml file:

`conda env create -f environment.yaml`

## Running

To reproduce experimental results of scICL on PBMC-10K dataset, you can run 

`python test_with_checkpoint.py -d '0' --config 'pbmc_10k' --checkpoint path_to_checkpoint`

The PBMC-10k-dataset can be download  from [here](https://scglue.readthedocs.io/zh-cn/latest/data.html)

The pretrained checkpoint can be download from [here](https://pan.quark.cn/s/52bfb960f8fe)

To train a scICL model, you can run

'python main.py -d '0' --config config_name --module 'pretrain''
'python main.py -d '0' --config config_name --module 'contrastive_learning '

