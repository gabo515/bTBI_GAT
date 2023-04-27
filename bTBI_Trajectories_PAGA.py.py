#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
import numba
import anndata
from scipy import sparse
import os,time,datetime,sys,pickle
from sklearn.model_selection import train_test_split


# In[2]:


import umap
from sklearn.manifold import TSNE

from scipy import stats as st
from collections import OrderedDict
from collections import defaultdict
from collections import Counter
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sknetwork.clustering import Louvain

from scipy.io import mmread
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph

import tarfile as tar
import csv
import gzip
import os
import scipy.io


# In[3]:


sc.settings.verbosity = 3
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white')


# In[4]:


path1= '/Users/dsoler/Desktop/Stat ML for Genomics/TBI Project/bTBI 2'
path2= '/Users/dsoler/Desktop/Stat ML for Genomics/TBI Project/Sham 2'

bTBI = sc.read_10x_mtx(path1, cache=True)
sham = sc.read_10x_mtx(path2, cache=True)


# In[5]:


bTBI.obs['bTBI']=True
sham.obs['bTBI']=False
bTBI.obs['condition'] = 'bTBI'
sham.obs['condition'] = 'sham'


# # 01. General Preprocessing #

# In[6]:


adata = bTBI.concatenate(sham)


# In[7]:


QC_metrics = sc.pp.calculate_qc_metrics(adata)


# In[8]:


cells_per_gene = QC_metrics[1]['n_cells_by_counts']
cells_per_gene = cells_per_gene[cells_per_gene > 0]
filtered_cells_per_gene = cells_per_gene[cells_per_gene > 20]
plt.figure(1)
histogram = plt.hist(np.log10(cells_per_gene), 150)
plt.figure(2)
histogram = plt.hist(np.log10(filtered_cells_per_gene), 150)


# In[9]:


sc.pp.filter_genes(adata, min_cells=20)
sc.pp.filter_cells(adata, min_genes=200)


# In[10]:


median_lib_size = np.median(QC_metrics[0]['total_counts'])
sc.pp.normalize_total(adata, target_sum=median_lib_size)
sc.pp.log1p(adata)


# In[11]:


sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata = adata[:, adata.var["highly_variable"]]


# # 01. Clustering #

# In[12]:


sc.pp.pca(adata, n_comps = 10, use_highly_variable=True, svd_solver='arpack') #fix order
sc.pp.neighbors(adata, knn=True)


# In[13]:


sc.tl.umap(adata)
sc.tl.leiden(adata)
sc.tl.louvain(adata)


# In[14]:


sc.settings.set_figure_params(figsize=(6,5))
sc.pl.umap(adata, color = ['batch','louvain', 'leiden'])


# In[15]:


sc.settings.set_figure_params(figsize=(6,5))
sc.pl.umap(adata, color = ['louvain'], legend_loc='on data', frameon = False, title = "UMAP - louvain clustering")


# # 03. Marker Genes #

# In[16]:


sc.tl.rank_genes_groups(adata, groupby='louvain', groups=['0'], method='wilcoxon') #paper uses this test
sc.pl.rank_genes_groups(adata, n_genes=19, sharey=False)
result = adata.uns['rank_genes_groups']
DEGs = result['names'].dtype.names
DEGs = result['names']


# In[17]:


print(pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(20))


# In[18]:


result = adata.uns['rank_genes_groups']
groups = result['names'].dtype.names
df = pd.DataFrame(
    {group + '_' + key[:1]: result[key][group]
    for group in groups for key in ['names', 'pvals']})

df


# In[19]:


sc.pl.rank_genes_groups_dotplot(adata, key='rank_genes_groups', n_genes=28)


# In[20]:


sc.tl.rank_genes_groups(adata, 'louvain', method='wilcoxon')
sc.tl.filter_rank_genes_groups(adata, min_in_group_fraction=0.25, min_fold_change=0.25)


# In[21]:


result = adata.uns['rank_genes_groups']
groups = result['names'].dtype.names
df = pd.DataFrame({group + '_' + key[:]: result[key][group]
                for group in groups for key in ['names', 'pvals_adj', 'logfoldchanges']})
df


# In[22]:


df[df['5_names']== 'Abcg2'] # this is expected so we do have NSCs in that cluster 


# In[23]:


df[df['5_names']== 'Vim'] # same here 


# In[24]:


df[df['16_names']== 'Abcg2'] # same here 


# In[25]:


df[df['5_names']== 'Vim'] #same here


# # 04. Neural Stem Cells # 
# ### Based on the literature Abcg2 and Vim are NSC marker genes #

# In[26]:


sc.pl.umap(adata, color=['Abcg2', 'Vim']) #looks like they are cluster 15


# In[27]:


# adata.obs['louvain_anno'].cat.categories = ['0', '1', '2', '3', '4', 'NSC', 'Neuron', '7', '8', '9', '10/Ery', '11', '12',
       #'13', '14', '15', '16', '17', '18', '19']


# In[28]:


labels_neuron = ['6','1', '8', '3', '0', '7', '4','19', '9', '11', '17']
labels_astrocyte = ['5','16']
labels_nsc =['15']
labels_microglia= ['10']
labels_oligodendrocyte = ['14','2']
labels_endothelial_mural = ['13']
labels_OPC = ['12']
labels_ependymal = ['18']


# In[29]:


adata.obs['Cells'] = ['Neuron' if i in labels_neuron
                             else 'Astrocyte' if i in labels_astrocyte
                             else 'NSC' if i in labels_nsc 
                             else 'Microglia' if i in labels_microglia
                             else 'Oligodendrocyte' if i in labels_oligodendrocyte
                             else 'Endothelial-Mural' if i in labels_endothelial_mural
                             else 'OPC' if i in labels_OPC
                             else 'Ependymal' if i in labels_ependymal
                             else 'Others' for i in adata.obs['louvain']]


# In[30]:


#adata.obs['louvain'] = adata.obs['Cells']


# In[31]:


sc.settings.set_figure_params(figsize=(6,6))

sc.pl.umap(adata, color=['louvain', 'batch', 'Cells'], 
        frameon=False, legend_loc='on data')


# In[32]:


nsc = adata[adata.obs['Cells'] == 'NSC', :]


# In[33]:


sc.pl.umap(nsc, color=['Abcg2', 'Vim'], legend_loc='on data') #our stem cell thus root cells


# In[34]:


# astros = adata[adata.obs['Cells'] == 'Astrocyte_NSC', :]
# epen  = adata[adata.obs['Cells'] == 'Ependymal', :]
# opc = adata[adata.obs['Cells'] == 'OPC', :]
# oligo= adata[adata.obs['Cells'] == 'Oligodendrocyte', :]


# In[35]:


adata.obs


# In[36]:


print(np.flatnonzero(adata.obs['Cells']  == 'NSC')[68]) #very little stem cells for some reason


# In[37]:


adata.uns['iroot'] = np.flatnonzero(adata.obs['Cells']  == 'NSC')[10]


# # 02. PAGA Trajectories #
# Note: There are not many NSCs (which is usually the case). Now that we know what genes are active in other cells after a bTBI (DEG analysis done in another notebook) we want to the phate of the NSCs. Explore routtes such as NSC -->Astrocyte , NSC -->Neuron 

# In[38]:


sc.tl.paga(adata, groups='Cells')


# In[39]:


sc.pl.paga(adata, color=['Cells', 'Hexb', 'Gfap','Vim'])


# In[40]:


sc.tl.draw_graph(adata, init_pos='paga')


# In[41]:


sc.pl.draw_graph(adata, color=['Cells'], legend_loc='on data')


# In[42]:


adata.uns['iroot'] = adata.uns['iroot'] = np.flatnonzero(nsc.obs['Cells']  == 'NSC')[147]# nsc.obs_names[147]


# In[43]:


sc.tl.dpt(adata)


# In[44]:


sc.pl.draw_graph(adata, color=['Cells', 'dpt_pseudotime'], legend_loc='on data')


# In[ ]:




