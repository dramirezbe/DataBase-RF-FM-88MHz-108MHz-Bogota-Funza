import pandas as pd
import glob
from tqdm import tqdm
import re

def load_node_data(files):
    """
    Loads CSV files starting with 'Node' into a dictionary.

    Iterates through all matching CSV files in the current directory, 
    extracts the node number from the filename, and loads the data 
    into a Pandas DataFrame. Explicitly skips 'Node8.csv'.

    Returns:
        dict: A dictionary where keys are node numbers (str) and 
              values are Pandas DataFrames containing the CSV data.
    """
    node_data = {}

    for file in tqdm(files, desc="Loading CSVs"):
        match = re.search(r'Node(\d+)', file)
        
        if match:
            node_number = match.group(1)
            
            if node_number == '8':
                continue
                
            node_data[node_number] = pd.read_csv(file)
            
    return node_data

def clean_node_data(node_data, target_rows=104):
    """
    Cleans node DataFrames by removing columns with 0/NaN values 
    and truncating rows to a consistent target length.
    """
    for node, df in node_data.items():
        print(f"\n--- Node {node} ---")
        
        # 1. Handle columns with 0 or NaN
        df_na = df.replace(0, pd.NA)
        cols_to_drop = df_na.columns[df_na.isna().any()].tolist()
        
        if cols_to_drop:
            print(f"Found NaN sectors in cols ({', '.join(cols_to_drop)}), dropping...")
            cleaned_df = df.drop(columns=cols_to_drop)
        else:
            cleaned_df = df

        # 2. Handle row consistency
        initial_rows = cleaned_df.shape[0]
        if initial_rows > target_rows:
            dropped_rows = initial_rows - target_rows
            print(f"Found inconsistency in number rows, dropping {dropped_rows} row, before {initial_rows} after {target_rows}.")
            cleaned_df = cleaned_df.iloc[:target_rows]
            
        # Update dictionary
        node_data[node] = cleaned_df

    return node_data