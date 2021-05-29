
"""
Code to conduct data analysis of Seattle 911 call data.
"""

import pandas as pd
import geopandas as gpd
import torch
import torch.nn as nn
import FeatureExtractor as feat
from model import NN

def main():
    #create_data('SPD_call_data_5_20_2021.csv')
    df = pd.read_csv('./data/Call_Data_2018.csv')
    extractor = feat.FeatureExtractor('word2vec')
    embeddings_dict = {}
    for feat_type in feat.TYPE_FEATURES:
        print(feat_type)
        embed_idx, embedding = extractor.get_embeddings(df[feat_type])
        embeddings_dict[feat_type] = [embed_idx, embedding]
        
    data = SPDCallDataset('./data/Call_Data_2018.csv')
    sample_vect = extractor.transform(data[0:2][0], embeddings_dict)
    print(sample_vect.shape)
    model = NN([sample_vect.shape[1], 1000, 500, 100], embeddings_dict, extractor.transform)
    pred = model(sample_vect)

class SPDCallDataset(torch.utils.data.Dataset):
    """
    OBJECT DESCRIPTION
    Map-style PyTorch Dataset
    FIELDS
    file_path - str: combined file path and file name of the truncated and
        processed .csv file.
    data - pd.DataFrame: DataFrame of features.
    y - pd.Series, dtype=int: Series of response times in seconds.
    """
    def __init__(self, file_path):
        """
        BEHAVIOR
        Instantiates the dataset.
        PARAMETERS
        file_path - str: File path to the truncated and processed .csv file.
        RETURNS
        n/a
        """
        self.file_path = file_path
        date_cols = ['Original Time Queued', 'Arrived Time']
        data_2018 = pd.read_csv(file_path, parse_dates=date_cols)
        data_2018.reset_index(inplace=True)
        self.y = data_2018['response_time'].copy()
        self.data = data_2018.drop(labels='response_time', axis=1)

    def __getitem__(self, idx):
        """
        BEHAVIOR
        Returns a sample in the dataset.
        PARAMETERS
        idx - int: Index of the dataset to return.
        RETURNS
        tuple: The data features in the form of a pd.Series are the 0-index
            of the tuple and the target value in the form of an integer is
            the 1-index of the tuple.
        """
        return self.data.loc[idx], self.y.loc[idx]

    def __len__(self):
        """
        BEHAVIOR
        Provides the number of samples in the dataset.
        PARAMETERS
        RETURNS
        """
        return len(self.data)


def load_data(file_path):
    """
    BEHAVIOR
    Loads our cleaned dataset into working memory in the form of a Pandas
    DataFrame, and provides a train/val/test split.
    PARAMETERS
    file_path - str: File path to the cleaned .csv that was created by the
        'create_data' function defined below.
    RETURNS
    data - pd.DataFrame:
    partitions - Dict: Train/val/test partitions. The keys are strings:
        'train', 'val', and 'test', with the corresponding values as lists
        of integers indicating the corresponding indexes for that partition.
    """
    # TODO: instantiates dataframe in working memory and returns dictionary of
    #       partitions
    pass


def create_data(file_path):
    """
    BEHAVIOR
    Generates our base dataset from the source .csv in the form of a Pandas
    DataFrame and also saves it as a new .csv. Only provides data from 
    25 Jan 2018 to present because of the precinct change that went into
    effect on 24 Jan 2018. Converts time columns into datetime format and
    adds a column for response time (in seconds).
    PARAMETERS
    file_path - str: file path, including file name to the base .csv data.
    RETURNS
    data_2018 - pd.DataFrame: cleaned and processed dataset.
    """
    data = pd.read_csv(file_path)

    fmt_original = '%m/%d/%Y %H:%M:%S %p'
    fmt_arrived = '%b %d %Y %H:%M:%S:%f%p'
    data['Original Time Queued'] = pd.to_datetime(data['Original Time Queued'],
                                                  format=fmt_original)
    data['Arrived Time'] = pd.to_datetime(data['Arrived Time'],
                                          format=fmt_arrived)

    cutoff = pd.Timestamp(year=2018, month=1, day=25)
    data_2018 = data[data['Original Time Queued'] >= cutoff].copy()
    time_delta = data_2018['Arrived Time'] - data_2018['Original Time Queued']

    # translate the pandas timedelta dtype into an integer representation
    # in the unit of seconds and add to the DataFrame
    data_2018['response_time'] = time_delta.apply(lambda x: x.seconds)

    data_2018.reset_index(inplace=True)
    data_2018.to_csv(r'data/Call_Data_2018.csv')
    return data_2018


def plot_beats(data):
    """
    There are 5 precincts, each with a police station:
    1. North
    2. East
    3. South
    4. West
    5. Southwest
    The precincts are subdivided into smaller sectors. There are 17 total
    land sectors within the 5 precincts ('H' sector covers water around
    Seattle, described below). Each sector is further subdivided into 3 beats,
    making 3 * 17 = 51 beats in the city.

    There is one additional set of 3 beats, labeled H1, H2, and H3. H1
    indicates the water on the West side of Seattle; H2 indicates the water
    of Union Bay; and H3 denotes the area of Lake Washington that borders
    Seattle. The '99' beat seems to denote the northern and souther borders
    of the city.

    BEHAVIOR
    PARAMETERS
    data - pd.DataFrame: the 911 call dataset.
    RETURNS
    """
    beats = gpd.read_file(r'data/geo/Seattle_Police_Beats_2018-Present.shp')

    # join beats with call data; join on 'beat' for the beats data

    # pd.merge(beat
    pass


if __name__ == '__main__':
    main()
