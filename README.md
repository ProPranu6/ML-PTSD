## Passive Data Processing

This repository contains scripts and notebooks for processing and analyzing passive wearable data. The passive data includes various streams of data collected passively from users.

## Files Description

The repository contains the following main files:

1. **paths.py**: This file contains all the path variables that link to the source of various passive data files, including Fitbit files. It also defines the paths to store general statistical insights of the data, intermediate processing results, and final processed results.

2. **stats.py**: The file contains all the functions required for generating statistics and plots based on the passive data. It analyzes the data and provides insights into various metrics and trends.

3. **processing.py**: This script is responsible for processing the passive data. It performs processing in four stages:
    - Removing duplicates through duplicate ID entries.
    - Finding overlapping time ranges and resolving conflicts arising from them by identifying data types of individual streams and performing splitting operations for the time ranges and associated data.
    - Performing conceptual aggregation, aka day-wise aggregation, of the data by identifying the day boundaries, splitting that further into day half boundaries, and aggregating all the data within a day following the aggregation metrics defined in the `features_info.csv` file.
    - Splitting the conceptual aggregates present in day and day half CSV files across the users for better readability and loading.

4. **passive_data_processing_interactively.ipynb**: This notebook provides an interactive environment for running the processing scripts. It performs the same operations as `processing.py` but also allows for storing files on NAS and running interactively.

## Instructions to Run

1. **Install Required Packages**: Before running any scripts or notebooks, ensure you have all the necessary Python packages installed. You can install them by running the following command:
   
   ```
   pip install -r requirements.txt
   ```

2. **paths.py**: Run the file like a regular Python script. It will display all the paths stored and their corresponding labels.

3. **stats.py**: Run the file like a regular Python script. It will prompt the user to enter an output directory where all the results pertaining to the stats of the passive data can be saved locally.

4. **processing.py**: Run the file like a regular Python script. It will prompt the user to enter an output directory where all the intermediate and final processed passive data can be saved locally. Note that it may take about 60-90 minutes for the `processing.py` file to process all the passive data.

5. **passive_data_processing_interactively.ipynb**: Run the notebook in an interactive environment. It performs the same operations as `processing.py` but also allows for storing files on NAS and running interactively.
