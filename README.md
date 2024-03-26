# predictive-model-bridging
Code for a predictive model for bridging in English (using data from the GUM corpus). 

The python and R scripts included in the repository are described below. 

Rather than providing the original data (GUM) in this repository, we include the data files outputed from extract_features.py, which are read in by the two remaining scripts. 

## extract_features.py
Usage: 
```
python extract_features.py
```
This script creates train and dev datasets for bridging classification. The script reads in conllu data files from  data directory called /dep and outputs the following files:
- gum_entity_instances.csv - extracted entity information from GUM
- gum_entity_pairs.csv - bridging, coref, and regular entity pairs
- dev.csv - balanced dev data for bridging classification
- train.csv - balanced train data for bridging classification
- train_dev.csv - train and dev data in one file

## predictive_model.py

Usage: 
```
python predictive_model.py
```
This script trains a random forest classifier on the previously generated training data, makes and saves predictions on the dev data, and creates a number of graphics for precision and recall errors and the analysis of important features from the model.

## data_analysis.R

Script for association plots of important features for bridging environments according to the random forest model.