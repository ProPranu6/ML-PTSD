import pandas as pd
import os
from datetime import datetime
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
import re
import shutil
from copy import deepcopy as DC
from intervaltree import IntervalTree, Interval
from warnings import filterwarnings
filterwarnings('ignore')
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MinuteLocator, SecondLocator
from io import StringIO

from paths import *
from stats import _get_type_stats_agg

import time

# Function to convert seconds to hh:mm:ss format
def time_elapsed(seconds):
    # Convert seconds to time.struct_time object
    time_struct = time.gmtime(seconds)

    # Format the time.struct_time object as hh:mm:ss
    formatted_time = time.strftime("%H:%M:%S", time_struct)
    
    return formatted_time

def first_mode(series):
    """
    Computes the first mode (most frequently occurring value) in a given pandas Series.

    Parameters:
    -----------
    series : pandas.Series
        A pandas Series containing data.

    Returns:
    --------
    object or None
        The first mode value if it exists in the Series; otherwise, None.
    """
    modes = pd.Series.mode(series)
    return modes.iloc[0] if not modes.empty else None

def cv(series):
    """
    Calculates the coefficient of variation (CV) for a given pandas Series.

    The coefficient of variation is a measure of relative variability and is the ratio of the standard deviation to the mean.

    Parameters:
    -----------
    series : pandas.Series
        A pandas Series containing data.

    Returns:
    --------
    float
        The coefficient of variation (CV) value calculated as the standard deviation of the series divided by the mean of the series.
        A small constant (1e-7) is added to the denominator to avoid division by zero errors in case the mean is close to zero.
    """
    return series.std() / (series.mean() + 1e-7)

def unique(series):
    """
    Counts the number of unique elements in a given pandas Series.

    Parameters:
    -----------
    series : pandas.Series
        A pandas Series containing data.

    Returns:
    --------
    int
        The count of unique elements in the Series.
    """
    return len(series.unique())


def split_overlaps(df, val_type:str="str"):
    """
    Utility function to remove overlapping records (records with overlapping time ranges for a user) from a DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with 'startDate' and 'endDate' columns.
    - keep (str): Method for selecting records when overlapping ('first', 'last', 'largest', 'smallest').

    Returns:
    - pd.DataFrame: DataFrame with overlapping records removed.
    """

    #Try converting the datatimes from strings to datetime objects if they are not already formatted
    try:
        df[['startDate', 'endDate']] = df[['startDate', 'endDate']].apply(lambda x: [datetime.fromisoformat(xi) for xi in x])
    except:
        pass

    #Ensure that time ranges are atleast one microsecond long
    starts, ends = df['startDate'], df['endDate'] + timedelta(microseconds=1)*(df['startDate'] == df['endDate'])
    df.drop(columns=['startDate', 'endDate'], inplace=True)
    indices = df.index
   
    #Create an ordered tree of all the time ranges
    interval_tree = IntervalTree(Interval(start, end, (end.timestamp() - start.timestamp(), df.loc[index].to_dict())) for start, end, index in zip(starts, ends, indices))
    #Split the overlapping time ranges
    interval_tree.split_overlaps()
    
    
    new_rows = []
    for interval in interval_tree:
        start, end, data = interval
        denom, other= data
        other = DC(other)
        nume = end.timestamp() - start.timestamp()
        
        #Split the 'value' data of the overlapping parts according to their split fraction and the datatype of the datastream
        other['startDate'] = start
        other['endDate'] = end - timedelta(microseconds=1)*(end == start + timedelta(microseconds=1))
        if val_type == "num" : #float
            other['value'] = round(nume/denom, 6)*other['value']
        elif val_type == "obj":  #str
            other['value'] = other['value']
        elif val_type == "rate" or val_type == "rate_variance":  
            other['value'] = other['value']
        else:
            raise Exception(f"No such known {val_type} available")

        new_rows += [other]

        

    return pd.DataFrame(new_rows)


#Processing functions        
def remove_duplicates(src='path to original aggregate passive', out='path to removed duplicates aggregate passive'):
    """
    Remove duplicate records from processed passive data.

    Parameters:
    - src (str): Path to the source directory containing processed passive data.
    - out (str): Path to the output directory for storing data without duplicates.
    """
    features = os.listdir(src)

    try:
        shutil.rmtree(out)
    except:
        pass

    os.makedirs(out, exist_ok=True)

    #Iterate through each datastream, aka feature, and remove the duplicate entries
    for feature in features:
        file = os.path.join(src, feature)
        if os.path.isdir(file):
            continue 
        
        df = pd.read_csv(os.path.join(src, feature))
        df.drop(labels=['Unnamed: 0.1', 'Unnamed: 0'], inplace=True, axis='columns', errors='ignore')

        #Consider any repetition of recorded value for a particular user at a particular time as duplicate and all others as non-duplicates BECAUSE :

        # Few important Functional Dependencies in the Data
        # non-null id -> (uid, startDate, endDate)
        # non-null id -> value
        # non-null id -> sourceBundleId
        # non-null id -> (uid, startDate, endDate, value)
        # non-null id -> (uid, startDate, endDate, sourceBundleId)
        # non-null id -> (sourceBundleId, value)
        # (uid, startDate, endDate, sourceBundleId) -> value
        # (uid, startDate, endDate, sourceBundleId) -> non-null id
        # (uid, startDate, endDate, value) -> non-null id
        # (uid, startDate, endDate, value) -> sourceBundleId 


        # From the dependencies above removing duplicates in (uid, startDate, endDate, value) should remove all the conceptual duplicates as id and this combination are functionally dependent on each other
        not_duplicate = (df.loc[:, ['uid', 'value', 'startDate', 'endDate']].duplicated() != True)  
        df = df[not_duplicate]
        df.reset_index(drop=True, inplace=True)
        df.to_csv(f'{out}{feature}', index=False)
    
    return
    
def resolve_inconsistencies(src='path to processed passive', out='path to removed duplicates processed passive', nummethod='mean', objmethod='mode'):
    """
    Resolve inconsistencies (multiple recorded values for same user and time) in processed passive data based on specified methods, by first preprocessing with resolving overlaps.

    Parameters:
    - src (str): Path to the source directory containing processed passive data.
    - out (str): Path to the output directory for storing data with resolved inconsistencies.
    - nummethod (str): Method for resolving inconsistencies in numeric data ('mean', 'median', 'sum', 'mode').
    - objmethod (str): Method for resolving inconsistencies in object (non-numeric) data ('mode').
    """
    
    features = os.listdir(src)

    try:
        shutil.rmtree(out)
    except:
        pass

    os.makedirs(out, exist_ok=True)

    for feature in features:
        file = os.path.join(src, feature)
        if not(os.path.isfile(file)):
            continue 
        
        #Process only csv files except skip
        print(f"Reading file {file}")
        try:
            df = pd.read_csv(file)
        except : 
            continue
        
        #Convert Datetime data to datetime objects
        df[['startDate', 'endDate']] = df[['startDate', 'endDate']].apply(lambda x: [datetime.fromisoformat(xi) for xi in x])

        feature_name = re.search('_(.*)\.', feature).group(1)
        
        #Identify the type of the data being processed to determine the path of further processing 
        val_type, _, _ = _get_type_stats_agg(df['value'][0], stream=feature_name, feature='value') 
        val_type = re.search(':(\w*)', val_type).group(1)  

        #Split the time ranges and the values for each user if there are any overlapping time ranges
        df = df.groupby(['uid']).apply(split_overlaps, val_type=val_type)
        df.reset_index(drop=True, inplace=True)
        df[['startDate', 'endDate']] = df[['startDate', 'endDate']].apply(lambda x: [xi.to_pydatetime() for xi in x])
   
        #Identify the region of data that has come from same sourceBundleId, aka app ID, and that hasn't
        has_same_apps = df[['uid', 'startDate', 'endDate', 'sourceBundleId']].duplicated(keep=False)
        has_different_apps =  has_same_apps != True     

        #Perform resolution of overlapping time ranges in region of data that has come from different apps according to the value_type of the data
        if val_type == "obj": 
            if objmethod == "mode":
                D = df[has_different_apps].groupby(['uid', 'startDate', 'endDate'])['value'].apply(lambda x: pd.Series.mode(x)[0])
            else:
                raise Exception("No such specified resolution method available for 'object' types")
        else:  
            if nummethod == "mean":
                D = df[has_different_apps].groupby(['uid', 'startDate', 'endDate'])['value'].mean()
            elif nummethod == "median":
                D = df[has_different_apps].groupby(['uid', 'startDate', 'endDate'])['value'].median()
            elif nummethod == "sum":
                if val_type == "num":
                    D = df[has_different_apps].groupby(['uid', 'startDate', 'endDate'])['value'].sum()
                else:
                    raise Exception(f"Sum resolution method can't be applicable for {val_type} types")
            elif nummethod == "mode":
                D = df[has_different_apps].groupby(['uid', 'startDate', 'endDate'])['value'].apply(lambda x: pd.Series.mode(x)[0])
            else:
                raise Exception("No such specified resolution method available for numeric types")          

        
        different_apps = df[has_different_apps].merge(D, how='left', on=['uid', 'startDate', 'endDate'], suffixes=('_drop', None), )  #Replace inconsistencies by the specified method
        different_apps.drop(labels=['value_drop'], inplace=True, axis='columns', errors='ignore')
        different_apps = different_apps.drop_duplicates(['uid', 'startDate', 'endDate', 'value'])  #Consider any repetition of recorded value for a particular user at a particular time as duplicate
        
        #Retain overlapping time ranges in data coming from same apps, and merge the resolved different-apps' overlapping data
        df = pd.concat([df[has_same_apps], different_apps], ignore_index=True)
        df.to_csv(f'{out}{feature}', index=False)      

    return  
       



def conceptual_aggregate(src='passive-agg-data/', out='passive-concpt-agg-data/', day=True, day_boundary=18, save_day_half=True, study_span_days=84, skip_existing_results=False, clean_existing_results=False):
    """
    Aggregate time-series data at different conceptual levels (day, week, month).

    Parameters:
    - src (str): Source directory containing input time-series data files.
    - out (str): Output directory to store aggregated data.
    - day (bool): Flag to perform aggregation at the day level.
    - day_boundary (int): The boundary hour for splitting data into days. Default is 18 (6 PM).
    - save_day_half (bool): Flag to save day-half level aggregated data. Default is True.
    - study_span_days (int): Number of days to consider for the study span. Default is 84 days.
    - skip_existing_results (bool): Flag to skip the creation of aggregated files if they already exist. Default is False.
    - clean_existing_results (bool): Flag to clean existing output directory before aggregating. Default is False.

    Returns:
    None
    """
    
    # List all features in the source directory
    features = os.listdir(src)
    
    if clean_existing_results:
        try:
            shutil.rmtree(out)
        except:
            pass


    # Process each feature
    for feature in features:
        # Read the CSV file for the current feature
        file = os.path.join(src, feature)
        if not(os.path.isfile(file)):
            continue
        
        # Extract feature name from the file name
        feature_name = re.search('_(.*)\.', feature).group(1)


        if (os.path.exists(f'{out}{feature_name}/day.csv')) and (os.path.exists(f'{out}{feature_name}/day_stats.csv')) and skip_existing_results:
            print(f"Skipping creation of : {out}{feature_name}/day.csv and {out}{feature_name}/day_stats.csv")
            continue
        
        try:
            csv = pd.read_csv(file)
        except:
            continue
     
        
        # Convert 'startDate' and 'endDate' columns to datetime objects
        try:
            csv[['startDate', 'endDate']] = csv[['startDate', 'endDate']].apply(lambda x: [datetime.fromisoformat(xi) for xi in x]) 
        except:
            pass

 
        def has_crossed(sd, ed, timeboundary=18):
            """
            Check if the time range has crossed the given day boundary.
            """
            sdb, edb = datetime(sd.year, sd.month, sd.day, timeboundary, tzinfo=sd.tzinfo), datetime(ed.year, ed.month, ed.day, timeboundary, tzinfo=sd.tzinfo)  
            return not(sd<=sdb and ed<=sdb) and not(sd>=sdb and ed <=edb) and not(sd >=edb and ed >=edb) 
        
        def split_overbound(csv, timeboundary=18, val_type=None):
            """
            Split time ranges spanning two different days at the specified boundary.
            """
            #Identify region of data that is overbound
            overbound_df = csv[csv.apply(lambda x: has_crossed(x['startDate'], x['endDate'], timeboundary=timeboundary), axis='columns')]
            
            #Split overbound data 
            splits = []
            for rowid in overbound_df.index:
                timestart, timeend = overbound_df.loc[rowid, ['startDate', 'endDate']]

                if timestart < datetime(timestart.year, timestart.month, timestart.day, timeboundary, tzinfo=timestart.tzinfo):  #startday's timeboundary is the boundary of split
                    dayBound1 = datetime(year=timestart.year, month=timestart.month, day=timestart.day, hour=timeboundary, minute=0, second=0, tzinfo=timestart.tzinfo)
                    dayBound2 = datetime(year=timestart.year, month=timestart.month, day=timestart.day, hour=timeboundary, minute=0, second=0, tzinfo=timeend.tzinfo)

                    day1fraction = (dayBound1.timestamp() - timestart.timestamp())/(timeend.timestamp() - timestart.timestamp())
                    day2fraction = 1-day1fraction

                else :  #endday's timeboundary is the boundary of split
                    dayBound1 = datetime(year=timeend.year, month=timeend.month, day=timeend.day, hour=timeboundary, minute=0, second=0, tzinfo=timestart.tzinfo)
                    dayBound2 = datetime(year=timeend.year, month=timeend.month, day=timeend.day, hour=timeboundary, minute=0, second=0, tzinfo=timeend.tzinfo)

                    day1fraction = (dayBound1.timestamp() - timestart.timestamp())/(timeend.timestamp() - timestart.timestamp())
                    day2fraction = 1-day1fraction

                
                #Split the 'value' data according to the type of the value
                row1, row2 = DC(csv.loc[rowid]), DC(csv.loc[rowid])
                if val_type == "num":
                    row1[['endDate', 'value']] = [dayBound1, round(csv.loc[rowid]['value']*day1fraction, 2)]
                    row2[['startDate', 'value']] = [dayBound2, round(csv.loc[rowid]['value']*day2fraction, 2)]
                else:
                    row1[['endDate']] = [dayBound1]
                    row2[['startDate']] = [dayBound2]


                csv.drop(index=rowid, inplace=True)
                splits += [row1, row2]

            #Merge the processed overbound data with the original data
            split_df = pd.DataFrame(splits)
            csv = pd.concat([csv, split_df], ignore_index=True, axis='index') 

            return csv
        
        #Identify the type of the value being processed for spefic path of further processing
        val_type_full, stats, aggmethod = _get_type_stats_agg(csv['value'][0], stream=feature_name, feature='value') 
        val_type = re.search(':(\w*)', val_type_full).group(1)  
        
        #split overbounding at the boundary
        csv = split_overbound(csv, timeboundary=day_boundary, val_type=val_type) 

        #Adjust the start date of study, according to the time boundary
        def adjust_startdate(date_str, timeperiod=18):
            date = datetime.fromisoformat(date_str)
            if date.hour >= timeperiod:
                return datetime(date.year, date.month, date.day, timeperiod)
            else:
                date = date - timedelta(1)
                return datetime(date.year, date.month, date.day, timeperiod)
            
        # Calculate the onset date for each user after adjusting it according to the time boundary
        meta_data = pd.read_csv(path_to_meta_data + 'users.csv', usecols=['doc_id', 'join_date'])    
        meta_data.rename({'doc_id':'uid'}, axis='columns', inplace=True)
        meta_data.set_index('uid', drop=True, inplace=True)
        potential_startdates = meta_data['join_date'].apply(adjust_startdate, timeperiod=day_boundary)


        #Calculate the seconds elapsed from the start of the study for each time range to calculate the days elapsed
        user_onset = potential_startdates 
        diff = csv['endDate'].apply(lambda x: datetime.timestamp(x)).values - user_onset[csv['uid']].apply(lambda x: datetime.timestamp(x)).values

        # Aggregate at the day level
        if day :
            csv['day'] = diff//(3600*24)
            #Split the data in a day for a user into day-halves
            csv = csv.groupby(['uid','day'])[['startDate', 'endDate', 'value']].apply(split_overbound, timeboundary=(day_boundary+12)%24).droplevel(2).reset_index()  #split at half day boundaries
            
            #Deciding Logic for assigning day-half labels from time ranges
            dayhalf_cat_cond = lambda x: x.hour >=day_boundary or x.hour <(day_boundary+12)%24 if day_boundary >= (day_boundary+12)%24 else x.hour >=day_boundary and x.hour <(day_boundary+12)%24
            csv['day_half'] = csv['endDate'].apply(lambda x: 0 if dayhalf_cat_cond(x) else 1) 
            csv['duration'] = csv['endDate'] - csv['startDate']
            csv['duration'] = csv['duration'].apply(lambda x: pd.to_timedelta(x).seconds//60)  #convert duration to mins
            
            #Creating template to store aggregated data for each user, for each day and for each dayhalf
            uids = np.unique(csv['uid'])
            uidsf = uids.reshape((uids.shape[0], 1, 1))
            daysf = np.full((1, study_span_days, 1), "", dtype=object)
            days_partf = np.full((1, 1, 2), "", dtype=object)  #adding day halves part
            data_studyspand = {'uid':(uidsf + daysf + days_partf).ravel(), 'day':[d for d in range(study_span_days) for i in range(2)]*uids.shape[0], 'day_half':[0, 1]*study_span_days*uids.shape[0]}
            csv_studyspandays = pd.DataFrame(data_studyspand)
            csv = csv_studyspandays.merge(csv, how='left', on=['uid', 'day', 'day_half'])
            csv[['value', 'duration']].fillna(0, inplace=True)

            if day:
                # Perform aggregation at the day level

                # Calculate various statistics
                central_disperse_stats = csv.groupby(['uid', 'day'])[['value', 'duration']].describe(include='all')
                if val_type != 'obj':
                    shape_stats = csv.groupby(['uid', 'day'])[['value', 'duration']].agg({'value' : ['skew', pd.Series.kurt, cv], 'duration' : ['skew', pd.Series.kurt, cv]})
                else:
                    shape_stats = csv.groupby(['uid', 'day'])[['value', 'duration']].agg({'duration' : ['skew', pd.Series.kurt, cv]})
                
                total_stats = central_disperse_stats.join(shape_stats, how='left').sort_index(level=0, axis=1).dropna(axis=1, how='all')
                day_level_stats = pd.concat({'stats' : total_stats}, axis=1)

                #Aggregate data according to the value type of the data
                if val_type == "num": 
                    day_level_agg = pd.concat({'aggregate' : csv.groupby(['uid', 'day']).agg({'value' : ['sum'], 'duration' : ['sum']})}, axis=1)
                    day_level_data = day_level_agg.join(day_level_stats, how='left').sort_index(level=0, axis=1).dropna(axis=1, how='all')
                elif val_type == "rate":
                    day_level_agg = pd.concat({'aggregate' : csv.groupby(['uid', 'day']).agg({'value' : [first_mode], 'duration' : ['sum']})}, axis=1)
                    day_level_data = day_level_agg.join(day_level_stats, how='left').sort_index(level=0, axis=1).dropna(axis=1, how='all')
                elif val_type == 'rate_variance': 
                    day_level_agg = pd.concat({'aggregate' : csv.groupby(['uid', 'day']).agg({'value' : ['mean'], 'duration' : ['sum']})}, axis=1)
                    day_level_data = day_level_agg.join(day_level_stats, how='left').sort_index(level=0, axis=1).dropna(axis=1, how='all')
                else:
                    day_level_agg = pd.concat({'aggregate' : csv.groupby(['uid', 'day']).agg({'value' : [unique], 'duration' : ['sum']})}, axis=1)
                    day_level_data = day_level_agg.join(day_level_stats, how='left').sort_index(level=0, axis=1).dropna(axis=1, how='all')
                
                #Rename the columns at all levels for better readibility
                old_names = day_level_data.columns.get_level_values(1).unique()
                new_names = {k : f"{feature_name}-" + k for k in old_names}
                day_level_data.rename(columns=new_names, level=1, inplace=True)

                old_names = day_level_data.columns.get_level_values(2).unique()
                new_names = {k : f"observation(s)-" + k for k in old_names}
                day_level_data.rename(columns=new_names, level=2, inplace=True)

                # Create subdirectories in the output directory
                os.makedirs(f'{out}{feature_name}/', exist_ok=True)            
                    
                # Save day-level aggregated data
                print(f"Saving day files for {out}{feature_name}")
                day_level_data.replace(0, np.nan, inplace=True) #replacing all the 0s with blanks to ensure blank lines only for nill data
                day_level_data.to_csv(f'{out}{feature_name}/day.csv')


                if save_day_half:
                    # Perform aggregation at the day-half level

                    central_disperse_stats = csv.groupby(['uid', 'day', 'day_half'])[['value', 'duration']].describe(include='all')
                    if val_type != 'obj':
                        shape_stats = csv.groupby(['uid', 'day', 'day_half'])[['value', 'duration']].agg({'value' : ['skew', pd.Series.kurt, cv], 'duration' : ['skew', pd.Series.kurt, cv]})
                    else:
                        shape_stats = csv.groupby(['uid', 'day', 'day_half'])[['value', 'duration']].agg({'duration' : ['skew', pd.Series.kurt, cv]})
                    
                    total_stats = central_disperse_stats.join(shape_stats, how='left').sort_index(level=0, axis=1).dropna(axis=1, how='all')
                    dayhalf_level_stats = pd.concat({'stats' : total_stats}, axis=1)
                    
                    if val_type == "num" or val_type == 'rate_variance': #float
                        dayhalf_level_agg = pd.concat({'aggregate' : csv.groupby(['uid', 'day', 'day_half']).agg({'value' : ['sum'], 'duration' : ['sum']})}, axis=1)
                        dayhalf_level_data = dayhalf_level_agg.join(dayhalf_level_stats, how='left').sort_index(level=0, axis=1).dropna(axis=1, how='all')
                    elif val_type == "rate":
                        dayhalf_level_agg = pd.concat({'aggregate' : csv.groupby(['uid', 'day', 'day_half']).agg({'value' : [first_mode], 'duration' : ['sum']})}, axis=1)
                        dayhalf_level_data = dayhalf_level_agg.join(dayhalf_level_stats, how='left').sort_index(level=0, axis=1).dropna(axis=1, how='all')
                    else:
                        dayhalf_level_agg = pd.concat({'aggregate' : csv.groupby(['uid', 'day', 'day_half']).agg({'value' : [unique], 'duration' : ['sum']})}, axis=1)
                        dayhalf_level_data = dayhalf_level_agg.join(dayhalf_level_stats, how='left').sort_index(level=0, axis=1).dropna(axis=1, how='all')
                    
                    old_names = dayhalf_level_data.columns.get_level_values(1).unique()
                    new_names = {k : f"{feature_name}-" + k for k in old_names}
                    dayhalf_level_data.rename(columns=new_names, level=1, inplace=True)

                    old_names = dayhalf_level_data.columns.get_level_values(2).unique()
                    new_names = {k : f"observation(s)-" + k for k in old_names}
                    dayhalf_level_data.rename(columns=new_names, level=2, inplace=True)
                
                    # Create subdirectories in the output directory
                    os.makedirs(f'{out}{feature_name}/', exist_ok=True)            
                        
                    # Save day-level aggregated data
                    print(f"Saving dayhalf files for {out}{feature_name}")
                    dayhalf_level_data.replace(0, np.nan, inplace=True) #replacing all the 0s with blanks to ensure blank lines only for nill data
                    dayhalf_level_data.to_csv(f'{out}{feature_name}/dayhalf.csv')
    
    return


def split_conceptual_aggregate_for_users(src='path to conceptual aggregates userwide data', out='path to user wise data', clean_existing_results=False, skip_existing_results=True):
    """
    Split the aggregated data from conceptual aggregation into user-wise data.

    Parameters:
    - src (str): Source directory containing aggregated conceptual data.
    - out (str): Output directory to store user-wise data.
    - clean_existing_results (bool): Flag to clean existing results in the output directory.
    - skip_existing_results (bool): Flag to skip creation of files for existing results.

    Returns:
    None
    """
    # List all features in the source directory
    features = os.listdir(src)

    # Clean existing results in the output directory if specified
    if clean_existing_results:
        try:
            shutil.rmtree(out)
        except:
            pass

    # Process each feature
    for feature in features:
        # Try reading the day.csv file for the current feature
        try:
            df = pd.read_csv(os.path.join(src, feature, 'day.csv'), header=[0, 1, 2], index_col=[0,1], skipinitialspace=True)
        except:
            continue
        
        # Process each user for the current feature
        for uid in df.index.get_level_values(0).unique():
            feature_user_path = os.path.join(out, feature, f"{uid}")
            # Skip creation if the file already exists and skipping existing results is enabled
            if (os.path.exists(os.path.join(feature_user_path, "day.csv"))) and skip_existing_results:
                print(f"Skipping creation of : {os.path.join(feature_user_path, 'day.csv')}")
                continue

            # Create directory for the user if it doesn't exist
            os.makedirs(feature_user_path, exist_ok=True)
            idx = pd.IndexSlice
            
            # Extract relevant columns for the user and concatenate them
            value_column = df.loc[[uid], idx[['aggregate', 'stats'], f'{feature}-value', :]]
            duration_column = df.loc[[uid], idx['aggregate', f'{feature}-duration', 'observation(s)-sum']]
            filtered_df = pd.concat([value_column, duration_column], axis=1).sort_index(level=0, axis=1)

            # Save the filtered data for the user
            filtered_df.to_csv(os.path.join(feature_user_path, "day.csv"))

        # Try reading the dayhalf.csv file for the current feature
        try:
            df = pd.read_csv(os.path.join(src, feature, 'dayhalf.csv'), header=[0, 1, 2], index_col=[0, 1, 2], skipinitialspace=True)
        except:
            continue

        # Process each user for the current feature
        for uid in df.index.get_level_values(0).unique():
            feature_user_path = os.path.join(out, feature, f"{uid}")
            # Skip creation if the file already exists and skipping existing results is enabled
            if (os.path.exists(os.path.join(feature_user_path, "dayhalf.csv"))) and skip_existing_results:
                print(f"Skipping creation of : {os.path.join(feature_user_path, 'dayhalf.csv')}")
                continue

            # Create directory for the user if it doesn't exist
            os.makedirs(feature_user_path, exist_ok=True)
            idx = pd.IndexSlice

            # Extract relevant columns for the user and concatenate them
            value_column = df.loc[[uid], idx[['aggregate', 'stats'], f'{feature}-value', :]]
            duration_column = df.loc[[uid], idx['aggregate', f'{feature}-duration', 'observation(s)-sum']]
            filtered_df = pd.concat([value_column, duration_column], axis=1).sort_index(level=0, axis=1)

            # Save the filtered data for the user
            filtered_df.to_csv(os.path.join(feature_user_path, "dayhalf.csv"))
    
    return


if __name__ == '__main__':

    save_dir = input("Please enter the directory where the processed files should be stored (example : example/): ")
    save_unduplicated = save_dir+'unduplicated/'
    save_non_overlapping = save_dir+'non_overlapping/'
    save_conceptual_aggregate_usrwide = save_dir+'conceptual_aggregate/usrwide/'
    save_conceptual_aggregate_usrwise = save_dir+'conceptual_aggregate/usrwise/'
    os.makedirs(save_unduplicated, exist_ok=True)
    os.makedirs(save_non_overlapping, exist_ok=True)
    os.makedirs(save_conceptual_aggregate_usrwide, exist_ok=True)
    os.makedirs(save_conceptual_aggregate_usrwise, exist_ok=True)

    tic = time.time()
    print(f"Processing Files in {path_to_og_agg_passive} to REMOVE DUPLICATES and store in {save_unduplicated}")
    remove_duplicates(src=path_to_og_agg_passive, out=save_unduplicated)

    print(f"Processing Files in {save_unduplicated} to REMOVE OVERLAPS and store in {save_non_overlapping}")
    resolve_inconsistencies(src=save_unduplicated, out=save_non_overlapping, nummethod='mean', objmethod='mode')

    print(f"Processing Files in {save_non_overlapping} to CONCEPTUALLY AGGREGATE and store in {save_conceptual_aggregate_usrwide}")
    conceptual_aggregate(src=save_non_overlapping, out=save_conceptual_aggregate_usrwide, day=True, day_boundary=18, study_span_days=84, clean_existing_results=False, skip_existing_results=False, save_day_half=True)  #path_to_proc_concpt_agg_consist_passive

    print(f"Processing Files in {save_conceptual_aggregate_usrwide} to SPLIT INTO USERWISE CONCEPTUAL AGGREGATES and store in {save_conceptual_aggregate_usrwise}")
    split_conceptual_aggregate_for_users(src=save_conceptual_aggregate_usrwide, out=save_conceptual_aggregate_usrwise, clean_existing_results=False, skip_existing_results=False)
    toc = time.time()
    print(f"{time_elapsed(toc-tic)} seconds elapsed for the processing")

        