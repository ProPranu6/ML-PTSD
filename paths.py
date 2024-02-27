#Defining all the directories to source and target passive, fitbit and general-insights

# Path to directory that contains meta data about users
path_to_meta_data = '/Volumes/MAVERICK/master-download/'

# Path to general insights files
path_to_gen_insights = '/Volumes/MAVERICK-SCRIPT/root/processed-data/gen-insights/'

# Path to raw stream data of passive and fitbit files
path_to_og_passive = '/Volumes/MAVERICK/master-download/storage/passive_data/'
path_to_og_fitbit = '/Volumes/MAVERICK/master-download/storage/user_upload_fitbit/'

# Path to aggregated data of passive files
path_to_og_agg_passive = '/Volumes/MAVERICK/processed-data/passive-agg-data/'
path_to_og_usr_passive = '/Volumes/MAVERICK/processed-data/passive-user-data/'

# Path to aggregated data of fitbit files
path_to_proc_agg_fitbit = '/Volumes/MAVERICK-SCRIPT/root/processed-data/fitbit-data/fitbit-01-agg-data/'
path_to_proc_usr_fitbit = '/Volumes/MAVERICK-SCRIPT/root/processed-data/fitbit-data/fitbit-01-user-data/'

# Path to directory where unduplicated aggregated passive data files are to be stored
path_to_proc_agg_passive = '/Volumes/MAVERICK-Script/root/processed-data/passive-data/agg-data/'       

# Path to directory where consistent passive data files (overlapping-free time ranges), are to be stored
path_to_proc_agg_consist_passive = path_to_proc_agg_passive + 'consist-by-mean_mode/' 

# Path to directory where conceptual aggregates, aka day level aggregation, of passive data are to be stored, further distinguished by userwide and userwise sub directories 
path_to_proc_concpt_agg_consist_usrwide_passive = '/Volumes/MAVERICK-Script/root/processed-data/passive-data/concpt-agg-data/usrwide/' 
path_to_proc_concpt_agg_consist_usrwise_passive = '/Volumes/MAVERICK-Script/root/processed-data/passive-data/concpt-agg-data/usrwise/'

if __name__ == '__main__':

    # List of path variables and their corresponding string values
    paths = {
        'path_to_meta_data': path_to_meta_data,
        'path_to_gen_insights': path_to_gen_insights,
        'path_to_og_passive': path_to_og_passive,
        'path_to_og_fitbit': path_to_og_fitbit,
        'path_to_og_agg_passive': path_to_og_agg_passive,
        'path_to_og_usr_passive': path_to_og_usr_passive,
        'path_to_proc_agg_fitbit': path_to_proc_agg_fitbit,
        'path_to_proc_usr_fitbit': path_to_proc_usr_fitbit,
        'path_to_proc_agg_passive': path_to_proc_agg_passive,
        'path_to_proc_agg_consist_passive': path_to_proc_agg_consist_passive,
        'path_to_proc_concpt_agg_consist_usrwide_passive': path_to_proc_concpt_agg_consist_usrwide_passive,
        'path_to_proc_concpt_agg_consist_usrwise_passive': path_to_proc_concpt_agg_consist_usrwise_passive
    }
    
    
    # Print the path variables and their values
    print("\n")
    print("{:^150s}".format("PATH VARIABLES"))
    print("\n\n\n")
    for name, path in paths.items():
        print(f'{name}: {path}')
    print("\n\n\n")

