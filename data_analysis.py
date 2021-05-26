
"""
Code to conduct data analysis of Seattle 911 call data.
"""
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import shapefile


def main():
    beats = gpd.read_file(r'data/geo/Seattle_Police_Beats_2018-Present.shp')
    data_2018 = load_data(r'data/Call_Data_2018.csv')
    plot_beats(beats, data_2018)


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
    def __init__(self, idxs, file_path):
        """
        BEHAVIOR
        Instantiates the dataset partition.
        PARAMETERS
        idxs - 1D np.ndarray: Array of index values that will represent the
            partition.
        file_path - str: File path to the truncated and processed .csv file.
        RETURNS
        n/a
        """
        self.file_path = file_path
        date_cols = ['Original Time Queued', 'Arrived Time']
        data_2018 = pd.read_csv(file_path, parse_dates=date_cols)

        self.y = data_2018.loc[idxs, 'response_time'].copy()
        data_2018.drop(columns='response_time', inplace=True)
        self.data = data_2018.loc[idxs, :].copy()

    def __getitem__(self, idx):
        """
        BEHAVIOR
        Returns a sample in the dataset partition.
        PARAMETERS
        idx - int: Index of the dataset partition to return.
        RETURNS
        tuple: The data features in the form of a pd.Series is the 0-index
            of the tuple and the target value in the form of an integer is
            the 1-index of the tuple.
        """
        return self.data.iloc[idx], self.y.iloc[idx]

    def __len__(self):
        """
        BEHAVIOR
        Provides the number of samples in the dataset.
        PARAMETERS
        RETURNS
        """
        return len(self.data)


def gen_partition_idxs(file_path, pct_test=0.15, pct_val=0.15, seed=21):
    """
    BEHAVIOR
    Generates the indexes for each partition of the data.
    PARAMETERS
    file_path - str: File path to the cleaned .csv that was created by the
        'create_data' function defined below.
    RETURNS
    partitions - Dict: Train/val/test partitions. The keys are strings:
        'train', 'val', and 'test', with the corresponding values as 1D arrays
        of integers indicating the corresponding indexes for that partition.
    """
    default_rng = np.random.default_rng(seed=seed)
    data_2018 = pd.read_csv(file_path)

    num_samples = len(data_2018)
    num_test = int(math.floor(pct_test * num_samples))
    num_val = int(math.floor(pct_val * num_samples))

    partitions = {}
    all_idxs = np.arange(num_samples)
    partitions['test'] = default_rng.choice(all_idxs, size=num_test,
                                            replace=False)
    t_v_idxs = np.setdiff1d(all_idxs, partitions['test'], assume_unique=True)
    partitions['val'] = default_rng.choice(t_v_idxs, size=num_val,
                                           replace=False)
    partitions['train'] = np.setdiff1d(t_v_idxs, partitions['val'],
                                       assume_unique=True)
    return partitions


def load_data(file_path):
    """
    BEHAVIOR
    Loads the full dataset (post Jan 2018) into working memory in the form
    of a Pandas DataFrame for data analysis. Use the 'SPDCallDataset' class
    for neural network applications.
    PARAMETERS
    file_path - str: File path to the truncated and processed .csv file.
    RETURNS
    data_2018 - pd.DataFrame: Full dataset of post Jan 2018 call data.
    """
    date_cols = ['Original Time Queued', 'Arrived Time']
    data_2018 = pd.read_csv(file_path, parse_dates=date_cols)
    return data_2018


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

    data_2018.reset_index(drop=True, inplace=True)
    data_2018.to_csv(r'data/Call_Data_2018.csv')
    return data_2018


def plot_beats(beats, data):
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
    Plots the response time by SPD 'beat'.
    PARAMETERS
    beats - gpd.GeoDataFrame:
    data - pd.DataFrame: the 911 call dataset.
    RETURNS
    n/a
    """
    beats = gpd.read_file(r'data/geo/Seattle_Police_Beats_2018-Present.shp')
    hood_path = 'data/geo/Community_Reporting_Areas/' \
                + 'Community_Reporting_Areas.shp'
    hoods = gpd.read_file(hood_path)
    hoods.to_crs(epsg=4326, inplace=True)

    avg_response_time = data.groupby('Beat')['response_time'].mean()
    avg_response_time /= 60

    merged = pd.merge(beats, avg_response_time, left_on='beat', right_on='Beat')

    is_99 = merged['sector'] == '99'
    h = merged['sector'] == 'H'
    merged = merged[(~h) & (~is_99)]

    fig, ax = plt.subplots(1)

    merged.plot(column='response_time', ax=ax, legend=True)
    hoods.plot(ax=ax, edgecolor='red', alpha=0)
    plot_precinct_hq(ax)
    plt.title('Average response time for each Police Beat in minutes')

    plt.show()


def plot_precinct_hq(ax):
    precincts = gpd.read_file('data/geo/precinct_points.shp')
    precincts[precincts['precinct'] != 'HQ'].plot(color='blue', ax=ax)


def create_precinct_shp():
    precincts = pd.read_csv('data/geo/precinct_locs.csv')
    with shapefile.Writer('data/geo/precinct_points') as w:
        w.field('precinct', 'C')
        for i in range(len(precincts)):
            w.point(precincts.loc[i, 'longitude'], precincts.loc[i, 'latitude'])
            w.record(precincts.loc[i, 'precinct'])


if __name__ == '__main__':
    main()
