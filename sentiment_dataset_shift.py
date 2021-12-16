# import modules
from data import default_posneg_loader, default_emotions_loader
from distributions import (extrapolate_sentiment_distribution, 
    extrapolate_dataset_distribution, compute_bhattacharyya_distances)
import pandas as pd


def save_results(filename, indices, matrix): 
    # create DataFrame, save as CSV file
    data_df = pd.DataFrame(matrix, indices, indices)
    data_df.to_csv(filename)

def sentiment_dataset_shift(loader_function=None, dataset_filename=None, 
    sentiment_filename=None): 
    # load datasets 
    if loader_function is None: 
        print('Please specify loader function to load datasets, '
            + 'see README.md for more information')
        return 
    elif dataset_filename is None or sentiment_filename is None: 
        print('Please specify valid CSV filenames to save results '
            + 'for both dataset-level shift and sentiment-level shift')
    datasets = loader_function() 
    print('Loaded datasets')

    # extrapolate distributions corresponding to dataset 
    (dataset_indices, dataset_mean, 
        dataset_covar) = extrapolate_dataset_distribution(datasets)
    print('Extrapolated dataset distributions')
    
    # compute dataset level dataset shift 
    dataset_sds_matrix = compute_bhattacharyya_distances(
        dataset_indices, dataset_mean, dataset_covar)
    print('Computed dataset-level shift')

    # extrapolate distributions corresponding to dataset and sentiment
    (sentiment_indices, sentiment_mean, 
        sentiment_covar) = extrapolate_sentiment_distribution(datasets)
    print('Extrapolated dataset-sentiment distributions')
    
    # compute sentiment level dataset shift 
    sentiment_sds_matrix = compute_bhattacharyya_distances(
        sentiment_indices, sentiment_mean, sentiment_covar)
    print('Computed sentiment-level shift')

    # save results as CSV files 
    save_results(dataset_filename, dataset_indices, dataset_sds_matrix)
    save_results(sentiment_filename, sentiment_indices, sentiment_sds_matrix)
    print('Saved results in CSV files\n')


if __name__=='__main__':
    # specify arguments as tuples (loader function, filename for dataset-level 
    # shift results, and filename for sentiment-level shift results) for each run 
    # of the sentiment dataset shift pipeline 
    arguments_list = [
        (
            default_posneg_loader, 
            './default_posneg_dataset_sds.csv', 
            './default_posneg_sentiment_sds.csv'
        ), 
        (
            default_emotions_loader, 
            './default_emotions_dataset_sds.csv', 
            './default_emotions_sentiment_sds.csv'
        )
    ]

    # quantify dataset shift for the datasets
    for i in range(len(arguments_list)): 
        # get arguments 
        args = arguments_list[i]

        # run dataset shift pipeline
        print(f'Run {i + 1} of Sentiment Dataset Shift pipeline')
        sentiment_dataset_shift(args[0], args[1], args[2])
