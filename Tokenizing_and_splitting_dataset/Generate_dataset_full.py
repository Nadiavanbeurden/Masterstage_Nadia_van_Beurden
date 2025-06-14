import pandas as pd
import numpy as np
from collections import deque
from typing import Any, Tuple
import argparse 

parser = argparse.ArgumentParser(description='Binning data')
parser.add_argument('-n_bins', type=int, default=2, help='Number of bins (= Vocabulary size)')
parser.add_argument('-eps', type=float, default=0.001, help='Epsilon around zero')

args = parser.parse_args()

n_bins = args.n_bins
eps = args.eps

# Number of four vector components to include
num_params = 4

file_path = "Signal_and_background_data.csv"

df = pd.DataFrame(pd.read_csv(file_path))

process_labels = df['Event label']
# Remove eventlabel when converting vector components.
df = df.iloc[:,:-1]

def get_background_momenta(df: pd.DataFrame) -> np.ndarray[Any,Any]:
    """ 
    Takes a dataframe as imput and casts this to the right format to interpret. 
    The first 20 columns denote the particle types, MET and METphi, which we exclude and include later
    The other columns are the transverse momentum, energy, pseudorapidity and scattering angle (18 x 4)
    We obtain a matrix storing first all transverse momenta, then all energies etc. 
    """ 
    arr_background = df.to_numpy()
    arr_background_momenta = arr_background[:, 20:].reshape(df.shape[0], 18, 4)

    return arr_background_momenta

def generate_component_dataframes(df: pd.DataFrame) -> Tuple[list[pd.DataFrame], 
                                                             pd.DataFrame,
                                                             pd.DataFrame,
                                                             pd.DataFrame,
                                                             pd.DataFrame,
                                                             pd.DataFrame,
                                                             pd.DataFrame,
                                                             pd.DataFrame]:
    """ 
    Takes a dataframe as input and returns dataframes of each of the four vector components,
    a dataframe containing only the particles type and a list of all these dataframes.
    """
    arr_background_momenta = get_background_momenta(df)

    list_of_dfs = [] 

    # For _ in ids, pt, E, metE, metphi, eta, phi
    for _ in range (7): 
        if _ == 0: 
            # particle type
            list_of_dfs.append(df.iloc[:,0:18].astype(int))
        elif _ == 1:
            # MET
            list_of_dfs.append(df.iloc[:,18:19])
        elif _ == 2: 
            # METphi
            list_of_dfs.append(df.iloc[:,19:20])
        else:
            # pt, E, eta, phi
            list_of_dfs.append(pd.DataFrame(arr_background_momenta[:,:,_-3]))
        
    return list_of_dfs, list_of_dfs[0], list_of_dfs[1], list_of_dfs[2], list_of_dfs[3], list_of_dfs[4], list_of_dfs[5], list_of_dfs[6]

def generate_bin_edges(df: pd.DataFrame, n_bins: int, eps: float) -> list[float]:
    """ 
    Generates the equifrequent bin edges for a dataframe based on the number of bins
    """ 
    # Small variation in min, max to include these values when binning
    min_val = df.min().min()-eps
    max_val = df.max().max()+eps

    # Sort values
    values = df.values.flatten()
    sorted_values = np.sort(values)
    
    # Remove all zeros 
    values_non_zero = deque(x for x in sorted_values if x!= 0)

    # Compute bin edges
    bin_size = len(values_non_zero) // n_bins
    remainder = len(values_non_zero) % n_bins

    bin_edges = []
    start = 0 

    bin_edges.append(min_val)
    for bin in range (n_bins): 
        if bin < remainder:
            extra = 1
        else:
            extra = 0
        end = start + bin_size + extra
        bin_edges.append(values_non_zero[end-1])
        start = end 
    bin_edges.append(max_val)
    
    return bin_edges

def devide_into_bins(df: pd.DataFrame, n_bins: int, eps: float) -> Tuple[list[float], pd.DataFrame]: 
    """
    Generates a tokenized dataframe based on the number of bins 
    """
    # Generate bin edges and bin labels 
    bin_edges = generate_bin_edges(df, n_bins, eps)
    bin_labels = np.arange(1,len(bin_edges))
    
    # Apply bins and bin labels to dataframe
    df_custom_bins = df.apply(lambda x: pd.cut(x, bins=bin_edges, labels=bin_labels)).astype(int)
    # Cast all original zero's back to zero
    df_custom_bins[df == 0.] = 0.

    return bin_edges, df_custom_bins

def bin_all_background_data(list_of_dfs: list[pd.DataFrame], n_bins: int, eps:float) -> pd.DataFrame:
    """
    Takes the list of tokenized four vector components and combines this into a dataframe of tokenized 
    vectors of length num_params. Each param token is stored in a new column.
    """
    df_all_data_binned = np.zeros((len(list_of_dfs[0]), len(list_of_dfs[0].columns), num_params))
    # Adjust which components to use by changing df_all_data_binned[:,:,<index>] and selecting 
    # the right component dataframe by devide_into_bins(df_<component>, bins, eps)[1]
    
    df_all_data_binned[:,:,0] = df_ids #ids
    df_all_data_binned[:,:,2] = devide_into_bins(df_E, n_bins, eps)[1] #pt
    df_all_data_binned[:,:,1] = devide_into_bins(df_pt, n_bins, eps)[1] #pt
    df_all_data_binned[:,:,3] = devide_into_bins(df_eta, n_bins, eps)[1] #pt
    #df_all_data_binned[:,:,2] = devide_into_bins(df_missing_Et, n_bins, eps)[1] #missing Et
    #df_all_data_binned[:,:,4] = devide_into_bins(df_phi, n_bins, eps)[1]
    return df_all_data_binned

def generate_conversion_dataframe(df_binned : pd.DataFrame, list_of_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """ 
    Takes a binned dataframe as inputs. Each param token was stored as a new column. 
    Computes the number of unique rows and generates a dataframe of these unique rows. 
    We can read each row as a unique vector.
    """
    unique_vecors = np.unique(df_binned.reshape(-1, len(list_of_dfs)), axis=0)
    df_unique_vectors = pd.DataFrame(unique_vecors).astype(int)
    return df_unique_vectors

def generate_conversion_dict(df: pd.DataFrame) -> dict[str, int]:
    """
    Generates a dictionary to convert each unique vector to a corresponding integer token.
    """
    conversion_dict = {}
    for i in range (len(df)):
        # Store tuple of the vector components
        conversion_dict[tuple(df.iloc[i,0:5])] = i
    return conversion_dict

def generate_vectorized_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a dataframe as input and converts this to tokenized vectors
    """
    df_vectorized = pd.DataFrame()

    for particle in range (0,18):    
        df_vectorized[f'Vector_{particle}'] = tuple(df[:,particle,:])

    def array_to_tuple(arr):
        return tuple(arr)

    df_vectorized = df_vectorized.map(array_to_tuple)
    return df_vectorized

def generate_binned_df(conversion_dict: dict[str, int], df: pd.DataFrame) -> pd.DataFrame: 
    """
    Takes in a (tokenized) dataframe of vectors and returns a dataframe containing the vector tokens.
    """
    def lookup_tuple(tpl):
        return conversion_dict.get(tpl, None)

    df_vector_binned = df.map(lookup_tuple)

    return df_vector_binned

# Generate dataframes for each of the four vector components, particle type, MET and METphi
list_of_dfs, df_ids, df_missing_Et, df_missing_phi, df_E, df_pt, df_eta, df_phi = generate_component_dataframes(df)
names = ['ID', 'Missing E_t', 'Missing phi', 'E', 'P_t', 'eta', 'phi']

# Bin data and compute the unique vectors
df_all_data_binned = bin_all_background_data(list_of_dfs, n_bins, eps)
unique_vectors = np.unique(df_all_data_binned.reshape(-1, num_params), axis=0)
df_unique_vectors = pd.DataFrame(unique_vectors)

# Define the conversion dict for vector -> vector token
conversion_dict = generate_conversion_dict(df_unique_vectors)

# Generate vectorized dataframes
df_vectorized = generate_vectorized_df(df_all_data_binned)
df_vector_binned = generate_binned_df(conversion_dict, df_vectorized)

column_names = [f'Vector_{particle}' for particle in range (0,18)]


#### Only activate when metE at end ####
"""
bin_edges_metE = generate_bin_edges(df_missing_Et, n_bins, eps)
bin_labels_metE = np.arange(len(df_unique_vectors),len(df_unique_vectors)+len(bin_edges_metE)-1)

# Apply bins and bin labels to dataframe
df_custom_bins_metE = df_missing_Et.apply(lambda x: pd.cut(x, bins=bin_edges_metE, labels=bin_labels_metE)).astype(int)
# Cast all original zero's back to zero
df_custom_bins_metE[df_missing_Et == 0.] = 0.
df_custom_bins_metE

df_vector_binned = pd.concat([df_vector_binned, df_custom_bins_metE], axis=1)
column_names = np.append(column_names, 'MetE')

df_vector_binned.columns = column_names

"""

### Only activate when metphi at end ####
"""
# bin_edges_metphi = generate_bin_edges(df_missing_phi, n_bins, eps)
# bin_labels_metphi = np.arange(bin_labels_metE[-1]+1,bin_labels_metE[-1]+len(bin_edges_metphi))
# print(bin_labels_metphi)

# # Apply bins and bin labels to dataframe
# df_custom_bins_metphi = df_missing_phi.apply(lambda x: pd.cut(x, bins=bin_edges_metphi, labels=bin_labels_metphi)).astype(int)
# # Cast all original zero's back to zero
# df_custom_bins_metphi[df_missing_phi == 0.] = 0.
# df_custom_bins_metphi

# column_names = np.append(column_names, 'Metphi')

# df_vector_binned = pd.concat([df_vector_binned, df_custom_bins_metphi], axis=1)
# df_vector_binned.columns = column_names

"""

# Include event label again
df_vector_binned['Event label'] = process_labels

# Define signal dataset
df_signal = df_vector_binned[df_vector_binned['Event label'] == 1]
df_signal = df_signal.drop(columns = ['Event label'])

# Define backgroun dataset
df_backgr = df_vector_binned[df_vector_binned['Event label'] != 1]
df_backgr = df_backgr.drop(columns = ['Event label'])

# Store Signal and background datasets
df_signal.to_csv(f'binned_Signal_data_{n_bins}_bins.csv', index = False)
df_backgr.to_csv(f'binned_Background_data_{n_bins}_bins.csv', index = False)



