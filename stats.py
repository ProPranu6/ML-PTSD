from paths import *

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
import time
from matplotlib.dates import DateFormatter, MinuteLocator, SecondLocator
from io import StringIO

# Filter out warnings
filterwarnings('ignore')

# Importing necessary modules
import matplotlib
matplotlib.use('Agg')  # Using Agg backend for non-interactive plotting
from io import StringIO

# Importing paths module
from paths import *


# Function to convert seconds to hh:mm:ss format
def time_elapsed(seconds):
    # Convert seconds to time.struct_time object
    time_struct = time.gmtime(seconds)

    # Format the time.struct_time object as hh:mm:ss
    formatted_time = time.strftime("%H:%M:%S", time_struct)
    
    return formatted_time


# Stat functions
def _get_type_stats_agg(val, stream, feature):
    """
    Determine the type, summary statistics, and aggregation method for a given value.

    Parameters:
        val (object): The value to analyze.
        stream (str): The data stream associated with the value.
        feature (str): The feature associated with the value.

    Returns:
        tuple: A tuple containing the type name, summary statistics, and aggregation method.
    """

    type_space = 'unknown'
    if type(val) == str:
        type_space = 'str'
    elif (val is not np.nan):
        type_space = 'num'
    else:
        pass
    
    num_sum_stats = ['count', 'min', 'max', 'quantiles', 'std', 'mean', 'cv', 'skewness', 'kurtosis']
    num_agg = 'sum'
    obj_sum_stats = ['count', 'unique', 'top', 'freq']
    obj_agg = 'unique'
    
    if type_space != 'unknown':
        val = str(val)
        if 'heart_rate_variability' in stream:
            stats = num_sum_stats
            agg = 'mean'
            type_name = f'{type_space}:rate_variance'
        elif 'heart_rate' in stream:
            stats = num_sum_stats
            agg = 'mode'
            type_name = f'{type_space}:rate'
        elif 'vo2max' in stream:
            stats = num_sum_stats
            agg = 'mode'
            type_name = f'{type_space}:rate'
        elif (val.replace(',', '').replace('.', '').isnumeric()):
            stats = num_sum_stats
            agg = num_agg
            type_name = f'{type_space}:num'
        elif re.sub('[\W+_]', '', val).isalpha():   #can also be alphanumeric characters, potentially any string with atleast one word type character is considered a string
            stats = obj_sum_stats
            agg = obj_agg
            type_name = f'{type_space}:obj'
        else:
            stats = ['needs inspection']
            agg = 'needs inspection'
            type_name = 'unknown:unknown'
    else:
        stats = ['needs inspection']
        agg = 'needs inspection'
        type_name = 'unknown:unknown'

    return type_name, stats, agg

def get_users_dist_over_passive_fitbit(redo=True):
    """
    Retrieve data about users' distribution over passive and Fitbit data streams.

    Parameters:
        redo (bool, optional): Flag to indicate whether to redo the computation if the file already exists.

    Returns:
        pandas.DataFrame: A DataFrame containing users' distribution over passive and Fitbit data streams.
    """
    
    stat_file = os.path.join(path_to_gen_insights, 'user_dist_over_passive_fitbit.csv')
    if (os.path.exists(stat_file) and redo) or (not(os.path.exists(stat_file))):
        passive_users = set(os.listdir(path_to_og_passive))
        fitbit_users = set(os.listdir(path_to_og_fitbit))

        data = {'Users' : list(passive_users.union(fitbit_users)), 'Passive' : [], 'Fitbit' : []}
    
        for usr in data['Users']:
            data['Passive'].append(usr in passive_users)
            data['Fitbit'].append(usr in fitbit_users)
        
        data = pd.DataFrame(data)
        data.to_csv(stat_file, index=False)
        return data
    else:
       return pd.read_csv(stat_file)
    
def get_passive_features_union(redo=True):
    """
    Retrieve the union of features present in passive data streams.

    Parameters:
        redo (bool, optional): Flag to indicate whether to redo the computation if the file already exists.

    Returns:
        pandas.DataFrame: A DataFrame containing the union of features present in passive data streams.
    """

    stat_file = os.path.join(path_to_gen_insights, 'passive_features_union.csv')

    if (os.path.exists(stat_file) and redo) or (not(os.path.exists(stat_file))):
        features = os.listdir(path_to_og_agg_passive)
        features_users = pd.DataFrame()
        for feature in features:
            csv = pd.read_csv(os.path.join(path_to_og_agg_passive, feature))
            df = pd.DataFrame(np.unique(csv['uid']), columns=['user'])
            df.insert(0, 'feature', re.search('_(.*)\.', feature).group(1))
            features_users = pd.concat([features_users, df], ignore_index=True)

        features_users.to_csv(stat_file, index=False)
        return features_users
    else:
        return pd.read_csv(stat_file)
    
def get_passive_features_user_count(redo=True):
    """
    Retrieve the count of users for each passive feature.

    Parameters:
        redo (bool, optional): Flag to indicate whether to redo the computation if the file already exists.

    Returns:
        pandas.DataFrame: A DataFrame containing the count of users for each passive feature.
    """
    stat_file = os.path.join(path_to_gen_insights, 'passive_features_user_count.csv')

    if (os.path.exists(stat_file) and redo) or (not(os.path.exists(stat_file))):
        
        if os.path.exists(os.path.join(path_to_gen_insights, 'passive_features_union.csv')):
            
            features_users = pd.read_csv(os.path.join(path_to_gen_insights, 'passive_features_union.csv'))
            user_count = features_users.groupby(['feature'])['user'].count()
            user_count.name = "user_count"
            user_count  = pd.DataFrame(user_count).reset_index()
            user_count.to_csv(stat_file, index=False)
            
            return user_count

        else:
            raise Exception("passive_features_union.csv not found make sure to call `get_passive_features_union` first")
    
    else:
        return pd.read_csv(stat_file)

def get_features_info(redo=True):
    """
    Retrieve information about features from both passive and Fitbit data streams.

    Parameters:
        redo (bool, optional): Flag to indicate whether to redo the computation if the file already exists.

    Returns:
        pandas.DataFrame: A DataFrame containing information about features from both passive and Fitbit data streams.
    """

    stat_file = os.path.join(path_to_gen_insights, 'features_info.csv')

    if (os.path.exists(stat_file) and redo) or (not(os.path.exists(stat_file))):
        
        passive_stream_paths = [os.path.join(dirpath,f) for (dirpath, dirnames, filenames) in os.walk(path_to_og_agg_passive) for f in filenames]
        fitbit_stream_paths = [os.path.join(dirpath,f) for (dirpath, dirnames, filenames) in os.walk(path_to_proc_agg_fitbit) for f in filenames]
        #os.listdir(path_to_og_agg_passive)
        #os.listdir(path_to_proc_agg_fitbit)

        D = pd.DataFrame([], columns=['feature', 'datastream', 'passive?', 'aggregate_by', 'summary_stats', 'data_type'])
        for stream_path in passive_stream_paths:
           stream = re.search('/([^/]*\.\w*)', stream_path).group(1)
           try:
               data_points =  pd.read_csv(stream_path, keep_default_na=False, usecols=lambda x: x not in ['Unnamed: 0', 'Unnamed: 0.1', 'startDate', 'endDate', 'uid', 'id', 'source', 'sourceName', 'sourceBundleId', 'Date'] +  [f'Unnamed: {i}' for i in range(1, 10)]) #load a chunk:100 of data
           except:
               print(f"Skipping as passive stream {stream} is not in .csv")
               continue
           
           #data_point.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'startDate', 'endDate', 'uid', 'id', 'source', 'sourceName', 'sourceBundleId', 'Date'] + [f'Unnamed: {i}' for i in range(1, 10)], errors='ignore', inplace=True)
           if len(data_points) == 0:
               print(f"Skipping passive stream : {stream} data due to unavailable rows")
               continue
           
           data_points_wonull = data_points.dropna()
           try:
               data_point = data_points_wonull.iloc[0, :]
           except:
               data_point = data_points.iloc[0, :]
               
           features = data_point.index
           
           
           
           for feature in features:
                val = data_point[feature]
                type_name, stats, aggmethod = _get_type_stats_agg(val, stream, feature)
                D.loc[len(D)] = {'feature':'passive_'+feature, 'datastream':re.search('_(.*)\.\w*', stream).group(1), 'aggregate_by': aggmethod, 'summary_stats':"/".join(stats), 'data_type':type_name}
        
        for stream_path in fitbit_stream_paths:
           
           stream = re.search('/([^/]*\.\w*)', stream_path).group(1)
           try:
               data_points =  pd.read_csv(stream_path, keep_default_na=False, usecols=lambda x: x not in ['Unnamed: 0', 'Unnamed: 0.1', 'startDate', 'endDate', 'uid', 'id', 'source', 'sourceName', 'sourceBundleId', 'Date'] +  [f'Unnamed: {i}' for i in range(1, 10)])
           except:
               print(f"Skipping as fitbit stream {stream} is not in .csv")
               continue
           
           #data_point.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'startDate', 'endDate', 'uid', 'id', 'source', 'sourceName', 'sourceBundleId', 'Date'] +  [f'Unnamed: {i}' for i in range(1, 10)], errors='ignore', inplace=True)
           #data_point = data_points.dropna().iloc[0, :]

           if len(data_points) == 0:
               print(f"Skipping fitbit stream : {stream} data due to unavailable rows")
               continue
           
           data_points_wonull = data_points.dropna()
           try:
               data_point = data_points_wonull.iloc[0, :]
           except:
               data_point = data_points.iloc[0, :]

           features = data_point.index

           
           
           for feature in features:
                #val_type = type(data_point[feature])
                val = data_point[feature]
                type_name, stats, aggmethod = _get_type_stats_agg(val, stream, feature)
                    
                D.loc[len(D)] = {'feature':'fitbit_'+feature, 'datastream':re.search('(.*)_(.*)\.\w*', stream).group(1), 'aggregate_by':aggmethod, 'summary_stats':"/".join(stats), 'data_type':type_name}
        
        
        D.to_csv(stat_file, index=False)
        
        return D

    
    else:
        return pd.read_csv(stat_file)
    
def get_compressed_features_info(redo=True):
    """
    Retrieve compressed information about features from both passive and Fitbit data streams.

    Parameters:
        redo (bool, optional): Flag to indicate whether to redo the computation if the file already exists.

    Returns:
        pandas.DataFrame: A DataFrame containing compressed information about features from both passive and Fitbit data streams.
    """

    if redo or not(os.path.exists(path_to_gen_insights+'features_info_compr.csv')):
        fi = pd.read_csv(path_to_gen_insights+'features_info.csv')
        print("Before dropping units length :", len(fi))
        fi = fi[[ np.all([wd not in x for wd in ['unit', 'Start', 'End', 'start', 'end', 'date', 'Date', 'DATE', 'timestamp'] ]) for x in fi['feature']]]
        fi.reset_index(drop=True, inplace=True)
        print("After dropping units length :", len(fi))
        fi.to_csv(path_to_gen_insights+'features_info_compr.csv')
        return fi
    else:
        return pd.read_csv(path_to_gen_insights+'features_info_compr.csv')
     
def get_passive_user_days_of_study(redo=True):
    """
    Retrieve the number of days each passive user participated in the study.

    Parameters:
        redo (bool, optional): Flag to indicate whether to redo the computation if the file already exists.

    Returns:
        pandas.DataFrame: A DataFrame containing the number of days each passive user participated in the study.
    """

    stat_file = os.path.join(path_to_gen_insights, 'user_days_of_study.csv')
    if (os.path.exists(stat_file) and redo) or (not(os.path.exists(stat_file))):
        D = None
        files = os.listdir(path_to_proc_agg_consist_passive)
        for file in files:
            df = pd.read_csv(os.path.join(path_to_proc_agg_consist_passive, file))
            try:
                df['startDate'] = df['startDate'].apply(lambda x: datetime.fromisoformat(x))
            except:
                continue

            days_of_study = pd.DataFrame((df.groupby(['uid'])['startDate'].max() - df.groupby(['uid'])['startDate'].min()))
            days_of_study = days_of_study.rename(columns={'startDate':'Days of Study'}).reset_index()
            days_of_study['stream'] = re.search('_(.*)\.csv', file).group(1)

            if type(D) == type(None):
                D = days_of_study
            else:
                D = pd.merge(D, days_of_study, how='outer')

        D.to_csv(stat_file, index=False)
        return D
    else:
       return pd.read_csv(stat_file)
    
def get_fitbit_streams_union(redo=False):
    """
    Retrieve the union of Fitbit streams.

    Parameters:
        redo (bool, optional): Flag to indicate whether to redo the computation if the file already exists.

    Returns:
        pandas.DataFrame: A DataFrame containing the union of Fitbit streams.
    """
    stat_file = os.path.join(path_to_gen_insights, 'fitbit_streams_union.csv')
    if (os.path.exists(stat_file) and redo) or (not(os.path.exists(stat_file))):
        fitbit_stream_paths = [os.path.join(dirpath,f) for (dirpath, dirnames, filenames) in os.walk(path_to_proc_agg_fitbit) for f in filenames]
        streams_users = pd.DataFrame()
        for stream_path in fitbit_stream_paths:
            stream = re.search('/([^/]*\.\w*)', stream_path).group(1)
            try:
                stream_name = re.search('/([^/]*)_\w*\.\w*', stream_path).group(1)
            except:
                pass
            print("Reading stream : ", stream_name)
            try:
               try:
                data_points =  pd.read_csv(stream_path, keep_default_na=False, usecols= ['user_id'])
               except:
                   pass
               
               try:
                   data_points =  pd.read_csv(stream_path, keep_default_na=False, usecols= ['uid'])
               except:
                   pass
               
               col = data_points.columns[0]
            except:
               print(f"Skipping as fitbit stream {stream} is not in .csv")
               continue

            if len(data_points) == 0:
               print(f"Skipping fitbit stream : {stream} data due to unavailable rows")
               continue

            df = pd.DataFrame(np.unique(data_points[col]), columns=['user'])
            df.insert(0, 'feature', stream_name)
            streams_users = pd.concat([streams_users, df], ignore_index=True)

        streams_users.to_csv(stat_file, index=False)
        return streams_users
    else:
        return pd.read_csv(stat_file)
    
def get_fitbit_streams_user_count(redo=False):
    """
    Retrieve the count of users for each Fitbit stream.

    Parameters:
        redo (bool, optional): Flag to indicate whether to redo the computation if the file already exists.

    Returns:
        pandas.DataFrame: A DataFrame containing the count of users for each Fitbit stream.
    """

    stat_file = os.path.join(path_to_gen_insights, 'fitbit_streams_user_count.csv')

    if (os.path.exists(stat_file) and redo) or (not(os.path.exists(stat_file))):
        
        if os.path.exists(os.path.join(path_to_gen_insights, 'fitbit_streams_union.csv')):
            
            streams_users = pd.read_csv(os.path.join(path_to_gen_insights, 'fitbit_streams_union.csv'))
            user_count = streams_users.groupby(['feature'])['user'].apply(lambda x: len(np.unique(x)))
            user_count.name = "user_count"
            user_count  = pd.DataFrame(user_count).reset_index()
            user_count.to_csv(stat_file, index=False)
            
            return user_count

        else:
            raise Exception("fitbit_streams_union.csv not found make sure to call `get_fitbit_streams_union` first")
    
    else:
        return pd.read_csv(stat_file)
    
def get_passive_users_days_available_per_stream():
    """
    Retrieve the number of days each passive user data is available per stream.

    Returns:
        pandas.DataFrame: A DataFrame containing the number of days each passive user data is available per stream.
    """

    D = pd.DataFrame([], columns=['stream', 'uid', 'days_available'])
    streams = os.listdir(path_to_proc_concpt_agg_consist_usrwise_passive)
    for stream in streams:
        stream_folder = os.path.join(path_to_proc_concpt_agg_consist_usrwise_passive, stream)

        if os.path.isdir(stream_folder):

            for user in os.listdir(stream_folder):
                file = os.path.join(path_to_proc_concpt_agg_consist_usrwise_passive, stream, user, 'day.csv')
                
                if os.path.exists(file):
                    ndays = len(pd.read_csv(file, header=[0, 1, 2], index_col=[0, 1]).dropna(how='any'))
                    D.loc[len(D)] = {'stream':stream, 'uid':user, 'days_available':ndays}
        
    return D


#Plot functions
def timelines(y, xstart, xstop, color='b'):
    """
    Plot timelines at y from xstart to xstop with given color.

    Parameters:
        y (float or array-like): The y-coordinate(s) of the timeline(s).
        xstart (float or array-like): The start point(s) of the timeline(s).
        xstop (float or array-like): The stop point(s) of the timeline(s).
        color (str, optional): The color of the timelines.

    Returns:
        None
    """
    plt.hlines(y, xstart, xstop, color, lw=4)
    plt.vlines(xstart, y+0.03, y-0.03, color, lw=2)
    plt.vlines(xstop, y+0.03, y-0.03, color, lw=2)

def plot_time_ranges(path, uid, id_types='sourceBundleId'):
    """
    Plot time ranges for a given user ID.

    Parameters:
        path (str): The path to the data file.
        uid (str): The user ID.
        id_types (str, optional): The type of ID.

    Returns:
        None
    """

    plt.figure(figsize=(20,12))
    df = pd.read_csv(path)

    #9XG8qlLKeoLj8X2t7vob
    df[['startDate', 'endDate']] = df[['startDate', 'endDate']].apply(lambda x: [datetime.fromisoformat(xi) for xi in x])
    df = df[df['uid'] == uid]
    df = df.sort_values(by=['startDate']).iloc[:100]
    df = df.loc[np.random.choice(df.index, 25, replace=False)]
   
 

    id_types = df[id_types].unique()
    colors = np.random.choice(list(matplotlib.colors.XKCD_COLORS.keys()), len(id_types), replace=False)

    cap = ['index:'+ str(i) for i in range(len(df))]
    df.index = cap
    smallest_start, largest_stop = min(df['startDate']), max(df['endDate'])
    for idind, id in enumerate(id_types):    
        #cap, start, stop = range(len(df)), starts, ends #data['caption'], data['start'], data['stop']
        dft = df[df[id_types] == id]
        _, start, stop = _, dft['startDate'], dft['endDate']

        captions, unique_idx, caption_inv = np.unique(cap, 1, 1)

        #Build y values from the number of unique captions.
        y = (caption_inv + 1) / float(len(captions) + 1)
        timelines(y[[int(ind.replace('index:', '')) for ind in dft.index]], start, stop, colors[idind])


    ax = plt.gca()
    ax.xaxis_date()
    myFmt = DateFormatter('%m/%d %H:%M')
    ax.xaxis.set_major_formatter(myFmt)

    ax.xaxis.set_major_locator(MinuteLocator(interval=60*10)) # used to be SecondLocator(0, interval=20)

    #To adjust the xlimits a timedelta is needed.
    delta = (largest_stop - smallest_start)/10


    plt.yticks(y[unique_idx], captions)
    plt.ylim(0,1)
    plt.xlim(datetime(year=2023, month=2, day=19)-delta, datetime(year=2023, month=2, day=20)+delta)
    datelist = pd.date_range(datetime(smallest_start.year, smallest_start.month, smallest_start.day),datetime(largest_stop.year, largest_stop.month, largest_stop.day)).tolist()
    ax.vlines(datelist, 0, 1, color='k', linestyle='dashed', )
    plt.xlabel('Time')
    plt.show()

def _plot_save(x, path):
    """
    Helper function to plot and save data.

    Parameters:
        x (pandas.DataFrame): The DataFrame containing data.
        path (str): The path to save the plot.

    Returns:
        None
    """
    x.sort_values(['days_available']).plot(x='uid', y='days_available', kind='bar', figsize=(20, 12), title=x.loc[x.index[0], 'stream'])
    dirr = os.path.join(path, f"{x.loc[x.index[0], 'stream']}/")
    os.makedirs(dirr, exist_ok=True)
    plt.savefig(dirr+'plot.png')


if __name__ == '__main__':
    
    redo_mode = False
    save_dir = input("Please enter the directory where the stats should be stored (example : example/): ")
    tic = time.time()
    os.makedirs(save_dir, exist_ok=True)

    save_here = os.path.join(save_dir, 'features_info.csv')
    print(f"Saving Features-Info File to `{save_here}`...")
    cfi = get_compressed_features_info(redo=redo_mode)
    try:
        cfi.to_csv(save_here)
    except Exception as e:
        print(e)
    print("Done\n\n")

    save_here = os.path.join(save_dir, 'users_dist_over_passive_fitbit.csv')
    print(f"Saving Users-Distribution-Over-Passive-Fitbit File to `{save_here}`...")
    udopf = get_users_dist_over_passive_fitbit(redo=redo_mode)
    try:
        udopf.to_csv(save_here)
    except Exception as e:
        print(e)
    print("Done\n\n")
    
    save_here = os.path.join(save_dir, 'passive_features_union.csv')
    print(f"Saving Passive-Features-Union File to `{save_here}`...")
    pfu = get_passive_features_union(redo=redo_mode)
    try:
        pfu.to_csv(save_here)
    except Exception as e:
        print(e)
    print("Done\n\n")

    save_here = os.path.join(save_dir, 'passive_features_user_count.png')
    print(f"Saving Passive-Features-User-Count Plot to `{save_here}`...")
    pfuc = get_passive_features_user_count(redo=redo_mode)
    pfuc.sort_values(['user_count'], inplace=True)
    pfuc.plot(x='feature', y='user_count', kind='barh')
    try:
        plt.savefig(save_here, bbox_inches='tight')
    except Exception as e:
        print(e)
    print("Done\n\n")

    save_here = os.path.join(save_dir, 'passive_users_days_available_per_stream')
    print(f"Saving Passive-Users-Days-Availability-Per-Stream Plot to `{save_here}`...")
    pudaps = get_passive_users_days_available_per_stream()
    plt.figure(figsize=(100,12))
    pudaps.groupby(['stream']).apply(_plot_save, save_here)   
    print("Done\n\n") 
    
    
    save_here = os.path.join(save_dir, 'fitbit_streams_union.csv')
    print(f"Saving Fitbit-Streams-Union File to `{save_here}`...")
    fsu = get_fitbit_streams_union(redo=redo_mode)
    try:
        fsu.to_csv(save_here)
    except Exception as e:
        print(e)
    print("Done\n\n")

    save_here = os.path.join(save_dir, 'fitbit_streams_user_count.png')
    print(f"Saving Fitbit-Streams-User-Count Plot to `{save_here}`...")
    fsuc = get_fitbit_streams_user_count(redo=redo_mode)
    fsuc.sort_values(['user_count'], inplace=True)
    fsuc.plot(x='feature', y='user_count', kind='barh')
    try:
        plt.savefig(save_here, bbox_inches='tight')
    except Exception as e:
        print(e)
    print("Done\n\n")

    toc = time.time()
    print(f"{time_elapsed(toc-tic)} seconds elapsed for the processing")
