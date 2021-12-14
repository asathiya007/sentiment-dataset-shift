import numpy as np
from numpy.lib.npyio import load
from numpy.random.mtrand import sample
import pandas as pd
from data import default_emotions_loader, loadEmotionsDataset, loadCrowdFlowerDataset, loadTwitterDataset
from distributions import extrapolate_dataset_distribution, _compute_bhattacharyya_distance


# info = default_emotions_loader()




def obtainAlikeDatasets(data, percent):
    '''
    Function to sample two new datasets from the input dataset.

    Params: 
    data (pd.DataFrame) - Labeled dataset with vectorized text
    percent (float) - What percent of the original dataset the new sample set's size should be 

    Returns: 
    sampleOne (pd.DataFrame) - DataFrame sample 1
    sampleTwo (pd.DataFrame) - DataFrame sample 2
    '''
    #Obtain 2 random samples based of the percent
    sampleOne = data.sample(frac=percent)
    sampleTwo = data.sample(frac=percent)

    return sampleOne, sampleTwo

twitterSet = loadTwitterDataset()

def datasetSimilarityExperiments():
    print("Computing alike datasets...")
    alikeOne, alikeTwo = obtainAlikeDatasets(twitterSet, .05)
    print("Done.")
    print("Computing dataset distributions...")
    #Compute distance between these two datasets
    indices, vector_means, vector_covariances = extrapolate_dataset_distribution(alikeOne)
    indices2, vector_means2, vector_covariances2 = extrapolate_dataset_distribution(alikeTwo)
    print("Done.")
    print("Computing shift...")
    shiftAlike = _compute_bhattacharyya_distance(vector_means, vector_means2, vector_covariances, vector_covariances2)
    print("Alike Dataset Shift")
    print(shiftAlike)



datasetSimilarityExperiments()