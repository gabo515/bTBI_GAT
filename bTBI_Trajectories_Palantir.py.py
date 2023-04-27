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


# # 02. Clustering #

# In[12]:


sc.pp.pca(adata, n_comps = 10, use_highly_variable=True, svd_solver='arpack')
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


# In[83]:


df[df['1_names']== 'Gfap'] # cluster 1 with Sox10 this is expected so we do have NSCs in that cluster, matching genes 


# In[23]:


df[df['5_names']== 'Vim'] # same here 


# In[24]:


df[df['16_names']== 'Abcg2'] # same here 


# In[25]:


df[df['5_names']== 'Vim'] #same here


# # 04. Neural Stem Cells (NSCs) # 
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


# In[97]:


adata.obs['Cells'] = ['Neuron' if i in labels_neuron
                             else 'Astrocyte' if i in labels_astrocyte
                             else 'NSC' if i in labels_nsc 
                             else 'Microglia' if i in labels_microglia
                             else 'Oligodendrocyte' if i in labels_oligodendrocyte
                             else 'Endothelial-Mural' if i in labels_endothelial_mural
                             else 'OPC' if i in labels_OPC
                             else 'Epen' if i in labels_ependymal
                             else 'Others' for i in adata.obs['louvain']]


# In[98]:


#adata.obs['louvain'] = adata.obs['Cells']
get_ipython().run_line_magic('matplotlib', 'tk')


# In[99]:


sc.settings.set_figure_params(figsize=(6,6))

sc.pl.umap(adata, color=['louvain', 'Cells'], 
        frameon=False, legend_loc='on data')


# In[100]:


nsc = adata[adata.obs['Cells'] == 'NSC', :]
root_cell = nsc.obs_names[147]

nsc.obs['root'] = ['root_NSC' if i == root_cell in nsc.obs_names
                             else ' ' for i in nsc.obs_names]
nsc


# In[101]:


sc.pl.umap(nsc, color=['root'], legend_loc='on data') #plot with start cell


# In[102]:


sc.pl.umap(nsc, color=['Abcg2', 'Vim'], legend_loc='on data') #our stem cell thus root cells


# In[39]:


#print([x for x in np.sort(adata.obsm['X_umap'], axis=0)])


# In[40]:


# astros = adata[adata.obs['Cells'] == 'Astrocyte_NSC', :]
# epen  = adata[adata.obs['Cells'] == 'Ependymal', :]
# opc = adata[adata.obs['Cells'] == 'OPC', :]
# oligo= adata[adata.obs['Cells'] == 'Oligodendrocyte', :]


# In[41]:


adata.obs


# In[42]:


print(np.flatnonzero(adata.obs['Cells']  == 'NSC')[68]) #very little stem cells 


# # 05. Palantir #
# Note: There are not many NSCs (which is usually the case). Now that we know what genes are active in other cells after a bTBI (DEG analysis done in another notebook) we want to the see phate of the NSCs. Explore routes such as NSC --> Astrocyte , NSC --> Neuron , NSC --> Oligodendrocytes

# In[43]:


#sc.pp.neighbors(adata, n_neighbors=30, use_rep='X', method='gauss')


# In[44]:


import palantir
import seaborn as sns
import warnings

get_ipython().run_line_magic('matplotlib', 'inline')

sns.set_style('ticks')
plt.rcParams['figure.figsize'] = [4, 4]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['image.cmap'] = 'Spectral_r'
warnings.filterwarnings(action="ignore", module="matplotlib", message="findfont")


# In[73]:


pca_projections = pd.DataFrame(adata.obsm['X_pca'], index=adata.obs_names)
dm_res = palantir.utils.run_diffusion_maps(pca_projections, n_components=5)


# In[74]:


ms_data = palantir.utils.determine_multiscale_space(dm_res)


# In[75]:


sc.tl.umap(adata)


# In[76]:


sc.pl.embedding(adata, basis='umap')


# Creating a UMAP that only has the cell_types of interest (Astro, Neuron, Oligo)

# In[85]:


# neuron = adata[adata.obs['Cells'] == 'Neuron', :]
# oligo = adata[adata.obs['Cells'] == 'Oligodendrocyte', :]
# astro = adata[adata.obs['Cells'] == 'Astrocyte', :]


# In[84]:


#sc.pl.embedding(neuron, oligo, astro, basis='umap') trying to only plot these


# In[49]:


adata.layers['MAGIC_imputed_data'] = palantir.utils.run_magic_imputation(adata, dm_res)


# In[50]:


sc.pl.embedding(adata, basis='umap', layer='MAGIC_imputed_data',
               color=['Cntnap2', 'Vim', 'Abcg2']) #Cntnap2 is a Neuron marker gene while the other two are for NSCs


# In[51]:


umap = pd.DataFrame(adata.obsm['X_umap'], index=adata.obs_names)


# In[52]:


palantir.plot.plot_diffusion_components(umap, dm_res)


# In[53]:


#terminal states should be Neuron, Oligodendrocytes and Glia
neuron = adata[adata.obs['Cells'] == 'Neuron', :]
oligo = adata[adata.obs['Cells'] == 'Oligodendrocyte', :]
astro = adata[adata.obs['Cells'] == 'Astrocyte', :]


# In[54]:


sc.pl.umap(neuron, color=['Cntnap2'], legend_loc='on data')


# In[55]:


sc.pl.umap(oligo, color=['Mog'], legend_loc='on data')


# In[56]:


astro.obs_names[150] #getting cell id for termiinal states


# In[57]:


terminal_states = pd.Series(['Neuron', 'Oligodendrocyte', 'Astrocyte'],  #confused about series titles 
                           index=['AACCACAGTCCTGTTC-1-0', 'ACTACGACACACCTTC-1-0', 'ATTTCTGTCGGCCCAA-1-0'])


# In[58]:


root_cell = nsc.obs_names[147] #from above 
pr_res = palantir.core.run_palantir(ms_data, root_cell, num_waypoints=500, terminal_states=terminal_states.index) #terminal_states=terminal_states.index


# In[103]:


palantir.plot.plot_palantir_results(pr_res, umap) #remove islands and only focus on cells that come from NSCs


# In[87]:


cells = ['AACCACAGTCCTGTTC-1-0', 'ACTACGACACACCTTC-1-0', 'ATTTCTGTCGGCCCAA-1-0']
palantir.plot.plot_terminal_state_probs(pr_res, cells) 


# Compare all the maps, from dpt, GAT, Palantir (talk a lot about it in discussion) 

# In[ ]:




