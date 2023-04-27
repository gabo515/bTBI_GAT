# bTBI_GAT
Exploring the Pathogenesis of Blast-Related Traumatic Brain Injuries (bTBI) in the Subventricular Zone with GATs

# Abstract
The advent of single-cell RNA sequencing (scRNA-seq) has brought about a revolution in biological research by offering an impartial view of the diversity of cells within tissues. Despite being widely employed to gain insights into both healthy and pathological states, the utilization of scRNA-seq for disease diagnosis or prognostication has been rather limited. Graph Attention Networks (GATs) serve as a promising avenue to perform cell phate analysis as they have the ability to learn from original features and graph structures from snRNAseq data. In this project, we present a graph attention network for predicting the disease-state from single-cell data on a dataset of adult mice sub-ventriuclar zone (SVZ) after blast-related traumatic brain injury (bTBI), a type of non-contact traumatic brain injury that can be challeging to diagnose. Our GAT model trained on single cell data obtained from mice SVZ for a cohort of 6 mice, divided into a bTBI group and a Sham group, and a total of 15272 cells. The GAT was corroborated with normal trajectory analysis methods resulting in a 95% accuracy in prediction which allowed us to further use the learned graph attention model to get biological insight into the cell types and genes in the prediction. We envision that the development of this preliminary pipeline can serve as the start of a diagnosis and clinical tool to accurately identify bTBIs and facilitate treatment. 

# Replication

1. pip install all the requirements via 'pip install requirements.txt'

2. Please look at the bTBI_GAT_Training.ipynb for tutorial on how to replicate our results in your own environment 

  *note: cuda version could be different so check your own environment and if not use cpu

  **note: upload the 'BESTGATWeights.pkl' file to the model to get our top performing model

  ***note: if visualizing via the 'bTBI_GAT_Latent_Space.ipynb' also upload the genies file which contains a list of the top 2000 DEGs
