# Interpretable machine learning model for quantum materials: Understanding magnetic anisotropy at the atomic level
This repository contains the codebase and datasets associated with the manuscript, *Interpretable machine learning model for quantum materials: Understanding magnetic anisotropy at the atomic level*, enabling the reproduction of key results. 

## Structure
Most of the codebase is provided as notebooks, which include comments and references linking directly to specific parts of the manuscript and suplementary information.

* `data`: Contains the complete dataset used to train the machine learning models.
* `src`: Includes additional code imported by the notebooks.
* `txtoutput`: Stores all generated outputs. These file can be used to reproduce the figures and tables presented in the manuscript.

## Notebooks to Reproduce Results
The main directory includes the following notebooks:
* [CorrelationMatrix.ipynb](CorrelationMatrix.ipynb): Reproduces correlation matrices:
  * Figure 1: Spearman’s correlation matrix
  * Figure S1: Kendall’s correlation coefficients
  * Figure S2: Pearson’s correlation coefficients
* [TrainModels.ipynb](TrainModels.ipynb): Reproduces the main results in the manuscript:
  * Tables: Table 1, Table 2
  * Figures (from the generated output): Figure 2, Figure 3, Figure S3, and Figure S4
  * Feature selection
  * SOC energy predictions 
* [TrainModels_predict_PT.ipynb](TrainModels_predict_PT.ipynb): Reproduces additional experiments using second-order perturbation theory (PT) as the target variable instead of MAE:
  * Reproduces supplementary results, e.g., Figure S7
* [TrainModels_PT_as_feature.ipynb](TrainModels_PT_as_feature.ipynb): Explores the use of PT values as input features in the model:
  * This analysis corresponds to the discussion in lines 406-420 of the manuscript.
