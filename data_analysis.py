
"""
Code to conduct data analysis of Seattle 911 call data.
"""
import pandas as pd
import geopandas as gpd


def main():
    pass


def create_data(file_path):
    """
    BEHAVIOR
    Generates our base dataset from the source .csv in the form of a Pandas
    DataFrame. Only provides data from 25 Jan 2018 to present because of the
    precinct change that went into effect on 24 Jan 2018. Converts time
    columns into datetime format and adds a column for response time.
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
    data_2018['response_time'] = data_2018['Arrived Time'] \
        - data_2018['Original Time Queued']
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
