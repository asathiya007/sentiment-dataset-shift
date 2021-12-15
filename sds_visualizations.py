# import modules
import dataframe_image as dfi
import pandas as pd

def load_results(filename):
    # load data from CSV file as DataFrame 
    data_df = pd.read_csv(filename)
    data_df.set_index(['Unnamed: 0'], inplace=True)
    data_df.index.name = None

    # export image 
    dfi.export(data_df, filename[: -4] + '.png')

    # return data 
    return data_df 


if __name__=='__main__':
    # define CSV filenames for which to create images 
    filenames = [
        './default_posneg_dataset_sds.csv', 
        './default_posneg_sentiment_sds.csv', 
        './default_emotions_dataset_sds.csv', 
        './default_emotions_sentiment_sds.csv'
    ]

    # save each CSV as as PNG visualization 
    for filename in filenames: 
        load_results(filename) 
