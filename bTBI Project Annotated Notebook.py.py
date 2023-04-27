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


# In[ ]:


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


# # This notebook is kindoff like our draft notebook # 

# In[3]:


sc.settings.verbosity = 3
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white')


# In[4]:


os.getcwd()


# # 01. Read Singlecell files #

# In[5]:


path1= '/Users/dsoler/Desktop/Stat ML for Genomics/TBI Project/bTBI 2'
path2= '/Users/dsoler/Desktop/Stat ML for Genomics/TBI Project/Sham 2'

bTBI = sc.read_10x_mtx(path1, cache=True)
sham = sc.read_10x_mtx(path2, cache=True)


# In[6]:


bTBI.obs['bTBI']=True
sham.obs['bTBI']=False
bTBI.obs['condition'] = 'bTBI'
sham.obs['condition'] = 'sham'


# # 02. Concatenate adata #

# In[7]:


adata = bTBI.concatenate(sham)


# In[8]:


sc.pl.highest_expr_genes(adata, n_top = 20)


# # 03. Preprocessing. Filter cells (min genes=200) and genes (min cells = 3). Normalization by median library size #

# In[9]:


QC_metrics = sc.pp.calculate_qc_metrics(adata)


# In[10]:


cells_per_gene = QC_metrics[1]['n_cells_by_counts']
cells_per_gene = cells_per_gene[cells_per_gene > 0]
filtered_cells_per_gene = cells_per_gene[cells_per_gene > 20]
plt.figure(1)
histogram = plt.hist(np.log10(cells_per_gene), 150)
plt.figure(2)
histogram = plt.hist(np.log10(filtered_cells_per_gene), 150)


# In[11]:


sc.pp.filter_genes(adata, min_cells=20)
sc.pp.filter_cells(adata, min_genes=200)


# In[12]:


median_lib_size = np.median(QC_metrics[0]['total_counts'])
sc.pp.normalize_total(adata, target_sum=median_lib_size)
sc.pp.log1p(adata)


# # 04. Find Highly Variable Genes #

# In[13]:


sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata = adata[:, adata.var["highly_variable"]]


# In[30]:


#genies = list(adata.var_names)


# In[31]:


#file = open('genies.pkl', 'wb')
#pickle.dump(genies, file)
#file.close()


# # 04. PCA and UMAP #

# In[15]:


sc.pp.neighbors(adata, knn=True) #run twice for some reason and the next blocks to get the right graph
sc.pp.pca(adata, n_comps = 10, use_highly_variable=True, svd_solver='arpack')


# In[16]:


sc.tl.umap(adata)
sc.tl.leiden(adata)
sc.tl.louvain(adata)


# In[17]:


sc.settings.set_figure_params(figsize=(6,5))
sc.pl.umap(adata, color = ['batch','louvain'])


# In[19]:


sc.settings.set_figure_params(figsize=(6,5))
sc.pl.umap(adata, color = ['louvain'], legend_loc='on data', frameon = False, title = "UMAP - louvain clustering")


# These cells are found in the SVZ, therefore some cell types that we expect to have include Ependymal, Astrocyte-NSC, Microglia, Endothelial&Mural, Neuron, Oligodendrocyte, OPC. 
# 
# 

# # 05. Finding Top Genes #

# In[20]:


sc.tl.rank_genes_groups(adata, groupby='louvain', groups=['0'], method='wilcoxon') #paper uses this test
sc.pl.rank_genes_groups(adata, n_genes=19, sharey=False)
result = adata.uns['rank_genes_groups']
DEGs = result['names'].dtype.names
DEGs = result['names']


# In[21]:


print(pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(19))


# In[31]:


result = adata.uns['rank_genes_groups']
groups = result['names'].dtype.names
df = pd.DataFrame(
    {group + '_' + key[:1]: result[key][group]
    for group in groups for key in ['names', 'pvals']})

df


# In[54]:


df[df['0_n'] == 'Gfap']


# In[57]:


sc.pl.umap(adata, color=['Cntnap2']) #'Abcg2', 'Vim', 'Dcx',


# # 06. Finding Marker genes. Plot dotplots and get dataframe #

# In[42]:


sc.pl.rank_genes_groups_dotplot(adata, key='rank_genes_groups', n_genes=28)


# In[43]:


sc.tl.rank_genes_groups(adata, 'louvain', method='wilcoxon')
sc.tl.filter_rank_genes_groups(adata, min_in_group_fraction=0.25, min_fold_change=0.25)


# In[44]:


result = adata.uns['rank_genes_groups']
groups = result['names'].dtype.names
df = pd.DataFrame({group + '_' + key[:]: result[key][group]
                for group in groups for key in ['names', 'pvals_adj', 'logfoldchanges']})
df_level1 = df


# In[99]:


df


# In[120]:


sc.pl.umap(adata, color=['Abcg2'])


# In[45]:


sc.pl.rank_genes_groups_dotplot(adata, key='rank_genes_groups_filtered',  n_genes=5)


# In[46]:


sc.pl.rank_genes_groups(adata, n_genes=20, sharey=False)


# In[48]:


sc.tl.rank_genes_groups(adata, 'louvain', method='wilcoxon')
sc.tl.filter_rank_genes_groups(adata, min_in_group_fraction=0.25, min_fold_change=0.25)


# In[49]:


sc.pl.rank_genes_groups_dotplot(adata, key='rank_genes_groups_filtered', n_genes=2)


# In[110]:


pd.DataFrame(adata.uns['rank_genes_groups']['names'])


# In[51]:


adata.uns['rank_genes_groups'].keys()


# In[109]:


result = adata.uns['rank_genes_groups']
groups = result['names'].dtype.names
df = pd.DataFrame({group + '_' + key[:]: result[key][group]
                for group in groups for key in ['names', 'pvals_adj', 'logfoldchanges']})
df


# # 07. Cell Type Annotation #

# In[53]:


labels_neuron = ['6','1', '8', '3', '0', '7', '4','19','18', '9', '11', '17']
labels_astrocyte_nsc = ['5','16']
labels_microglia= ['10']
labels_oligodendrocyte = ['14','2']
labels_endothelial_mural = ['13']
labels_OPC = ['12']
labels_ependymal = ['15']


# In[54]:


adata.obs['Cells'] = ['Neuron' if i in labels_neuron
                            else 'Astrocyte_NSC' if i in labels_astrocyte_nsc 
                            else 'Microglia' if i in labels_microglia
                            else 'Oligodendrocyte' if i in labels_oligodendrocyte
                            else 'Endothelial-Mural' if i in labels_endothelial_mural
                            else 'OPC' if i in labels_OPC
                            else 'Ependymal' if i in labels_ependymal
                            else 'Others' for i in adata.obs['louvain']]


# In[55]:


sc.settings.set_figure_params(figsize=(6,6))

sc.pl.umap(adata, color=['louvain', 'batch', 'Cells'], 
        frameon=False, legend_loc='on data')


# # 08. KNN #

# In[23]:


X_pca = adata.obsm['X_pca']
kmeans = KMeans(n_clusters=19).fit(X_pca)
adata.obs['Kmeans'] = kmeans.labels_
sc.pl.umap(adata, color='Kmeans')


# In[56]:


sc.pp.neighbors(adata, n_neighbors=30,n_pcs=10, metric='cosine') # based on the kmeans value
#sc.tl.umap(adata)
sc.pl.umap(adata, edges=True, edges_color='purple')


# # 09. Trajectories Attempt #1 with PAGA and Zheng 2017 #
# 

# In[57]:


from matplotlib import rcParams 


# In[58]:


sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_versions()
results_file = './write/paul15.h5ad'
sc.settings.set_figure_params(dpi=80, frameon=False, figsize=(6, 5), facecolor='white')  # low dpi (dots per inch) yields small inline figures


# In[59]:


adata


# In[60]:


sc.pp.recipe_zheng17(adata)


# In[61]:


sc.tl.pca(adata, svd_solver='arpack') #try with n_comps =10


# In[62]:


sc.pp.neighbors(adata, n_neighbors=4, n_pcs=20)
sc.tl.draw_graph(adata)


# In[63]:


sc.pl.draw_graph(adata, color='Cells', legend_loc='on data')


# In[64]:


sc.tl.diffmap(adata)
sc.pp.neighbors(adata, n_neighbors=10, use_rep='X_diffmap')


# In[67]:


sc.tl.louvain(adata, resolution=1.0)


# In[68]:


sc.tl.paga(adata, groups='Cells')
#sc.pl.umap(adata, color=['Cntnap2','Scl1a2']


# In[69]:


sc.pl.paga(adata, color=['Cells'])


# # 10. Looking at Astrocytes-NSC and Ependymal Cells #

# In[70]:


sc.settings.verbosity = 3
sc.tl.draw_graph(adata, init_pos='paga')


# In[71]:


astros = adata[adata.obs['Cells'] == 'Astrocyte_NSC', :]
epen  = adata[adata.obs['Cells'] == 'Ependymal', :]
opc = adata[adata.obs['Cells'] == 'OPC', :]
oligo= adata[adata.obs['Cells'] == 'Oligodendrocyte', :]


# In[72]:


sc.pl.draw_graph(astros, color=['Cells'], legend_loc='on data')


# In[79]:


astros.obs_names[150] #root cell ID, any cell in the Astrocycte cluster works 


# In[73]:


sc.pl.draw_graph(epen, color=['Cells'], legend_loc='on data')


# In[74]:


sc.pl.draw_graph(oligo, color=['Cells'], legend_loc='on data')


# In[92]:


#adata.uns['iroot'] = np.flatnonzero(adata.obs['Cells'] == 'Astrocyte_NSC')


# # 10. sc.diff and sc.dpt # 

# In[82]:


sc.pp.neighbors(adata, n_neighbors=30, use_rep='X', method='gauss')


# In[83]:


sc.tl.diffmap(adata)


# In[89]:


astros


# In[90]:


sc.tl.dpt(adata, n_branchings=1, n_dcs=10)
astros.uns['iroot'] = 101
astros.var['xroot'] = astros['ATTTCTGTCGGCCCAA-1-0'].X #root_cell_name


# In[53]:


sc.pl.diffmap(adata, color=['Cells', 'dpt_pseudotime'])


# # 11. Diffusion Maps and Palantir. Note: we decided to proceed with Diffusion analysis with Palantir # 

# Person correlation with all the genes and get a scores, which genes have the most similar trajectory (GSA for pathways)
# 
# Palentir (Github)

# In[ ]:





# In[91]:


import palantir
import warnings 
warnings.filterwarnings('ignore')


# In[92]:


get_ipython().run_line_magic('matplotlib', 'inline')

sns.set_style('ticks')
plt.rcParams['figure.figsize'] = [4, 4]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['image.cmap'] = 'Spectral_r'


# In[93]:


pca_projections = pd.DataFrame(adata.obsm['X_pca'], index=adata.obs_names)
dm_res = palantir.utils.run_diffusion_maps(pca_projections, n_components=5)


# In[94]:


ms_data = palantir.utils.determine_multiscale_space(dm_res)


# In[97]:


sc.pp.neighbors(adata, n_neighbors=30,n_pcs=10, metric='cosine')
sc.tl.umap(adata) 


# In[98]:


sc.pl.embedding(adata, basis='umap')


# In[49]:


adata.layers['MAGIC_imputed_data'] = palantir.utils.run_magic_imputation(adata, dm_res)


# In[50]:


sc.pl.embedding(adata, basis='umap', layer='MAGIC_imputed_data',
               color=['Cntnap2']) #testing on Cntnap2 and then testing on the other marker genes that were idetified, find specific stemcell marker gene


# In[56]:


umap = pd.DataFrame(adata.obsm['X_umap'], index=adata.obs_names) #dpt module 


# In[72]:


astros


# In[59]:


palantir.plot.plot_diffusion_components(umap, dm_res) #component 4 seems interesting


# In[99]:


sc.tl.diffmap(adata)


# In[67]:


sc.tl.dpt(adata, n_dcs=10, n_branchings=1)


# In[68]:


sc.pl.diffmap(adata, color=['dpt_pseudotime', 'dpt_groups'])


# In[85]:


start_cluster = adata[adata.obs['Cells'] == 'Neuron'[1], :]


# In[86]:


sc.pl.draw_graph(start_cluster, color=['Cells'], legend_loc='on data')


# In[76]:


QC_metrics[0]


# In[87]:


adata


# In[ ]:
# In[ ]:

# # Attempt at Monnocle but rpy2 is annoying #

# In[ ]:


# import gseapy 
# import re
# import shutil

# #settings
# sc.settings.set_figure_params(dpi=200, facecolor='white')
# sc.settings.verbosity = 3

# #Data Preparation

# adata.layers["counts"] = adata.X.copy()
# data_mat_mon = adata.layers['counts'].T
# var_mon=adata.var.copy()
# obs_mon=adata.obs.copy()


# In[ ]:


# # R libraries
# import rpy2.robjects as robjects
# from rpy2.robjects.packages import importr
# from rpy2.robjects import numpy2ri
# from scipy.sparse import csc_matrix

# import rpy2.rinterface_lib.callbacks
# import logging

# from rpy2.robjects import pandas2ri
# import anndata2ri
# from gprofiler import gprofiler

# %load_ext rpy2.ipython
# #%reload_ext rpy2.ipython
# %matplotlib inline
# pandas2ri.activate()
# anndata2ri.activate()
# # Convert the dense NumPy array to an R object
# numpy2ri.activate()


# In[ ]:


# library(scran)
# library(RColorBrewer)
# library(slingshot)
# library(monocle)
# library(gam)
# library(clusterExperiment)
# library(ggplot2)
# library(plyr)
# library(MAST)
# library(Seurat)
# library(tidyverse)
# library(magrittr)


# In[ ]:


# data_mat_mon = adata.layers['counts'].T
# # var_mon=adata.var.copy()
# obs_mon=adata.obs.copy()


# In[ ]:


# %%R -i data_mat_mon -i obs_mon -i var_mon
# pd <- AnnotatedDataFrame(data = obs_mon)
# # genes
# fd <- AnnotatedDataFrame(data = var_mon)
# # cell barcodes as columns of matrix
# colnames(data_mat_mon) <- rownames(pd)
# # genes as rows of matrix
# rownames(data_mat_mon) <- rownames(fd)
# ie_regions_cds <- newCellDataSet(cellData=data_mat_mon, phenoData=pd, featureData=fd, expressionFamily=negbinomial.size())


# In[ ]:


# #Normalize the count data
# ie_regions_cds <- estimateSizeFactors(ie_regions_cds)


# In[ ]:


# #Calculate dispersions to filter for highly variable genes
# ie_regions_cds <- estimateDispersions(ie_regions_cds)


# In[ ]:


# #Do dimensionality reduction
# ie_regions_cds <- reduceDimension(ie_regions_cds, norm_method = 'vstExprs', reduction_method='DDRTree', verbose = T)


# In[ ]:


# #Run for the first time to get the ordering
# ie_regions_cds <- orderCells(ie_regions_cds)


# In[ ]:


# #Get a nice colour map
# custom_colour_map = brewer.pal(length(unique(pData(ie_regions_cds)$new_cluster)),'Paired')


# In[ ]:


# #Find the correct root state that coresponds to the 'Stem' cluster
# tab1 <- table(pData(ie_regions_cds)$State, pData(ie_regions_cds)$new_cluster)
# id = which(colnames(tab1) == 'MSC')
# root_name = names(which.max(tab1[,id]))


# In[ ]:


# # Visualize with our cluster labels
# options(repr.plot.width=5, repr.plot.height=4)


# In[ ]:


# cell_trajectory <- plot_complex_cell_trajectory(ie_regions_cds, color_by = new_cluster, show_branch_points = T, 
#                              cell_size = 2, cell_link_size = 1, root_states = c(root_name)) +
# scale_size(range = c(0.2, 0.2)) +
# theme(legend.position="left", legend.title=element_blank(), legend.text=element_text(size=rel(1.5))) +
# guides(colour = guide_legend(override.aes = list(size=6))) + 
# scale_color_manual(values = custom_colour_map)
# cell_trajectory

