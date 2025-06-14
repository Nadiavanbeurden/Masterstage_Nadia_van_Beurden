##################################################################### Import packages ######################################################
import math
import random
import keras
import time
import h5py
import sys
import os 
import datetime
import gc
import argparse 
import os 

import numpy as np 
import pandas as pd

from pathlib import Path

os.environ['TF_USE_LEGACY_KERAS']="1" 

import tensorflow as tf
print(tf.test.is_built_with_cuda())

from Encoder import Encoder

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

##################################################################### Read parameters ######################################################


parser = argparse.ArgumentParser(description='Training the encoder model')
parser.add_argument('-test_path', type=str, default=None, help='Path to test set')
parser.add_argument('-min_length', type=int, default=5, help='Start length of features')
parser.add_argument('-num_iterations', type=int, default=1, help='Number of iterations to predict')
parser.add_argument('-weights_path', type=str, default=None, help='Path to model weights')
parser.add_argument('-history_path', type=str, default=None, help='Path to model history')
parser.add_argument('-store_path', type=str, default=None, help='Path to where predictions will be saved')
parser.add_argument('-dataset', type=str, default=None, help='signal or background')
args = parser.parse_args()

verbose = True

test_path = args.test_path
weights_path = args.weights_path
history_path = args.history_path 
store_path = args.store_path
num_iterations = args.num_iterations
min_length = args.min_length
dataset = args.dataset

df_test_data = pd.read_csv(test_path)
f = h5py.File(history_path, 'r')

# Define hyper parameters
vocab_size = f.attrs['vocab_size']
max_input_size = f.attrs['max_input_size']
num_heads = f.attrs['h']
num_layers = f.attrs['n']
d_model = f.attrs['d_model']
d_ff = f.attrs['d_ff']
drop_rate = f.attrs['drop_rate']
batch_size = f.attrs['batch_size']

num_batches = int(np.ceil(len(df_test_data)/batch_size))

##################################################################### Define functions ######################################################

def generate_batches(df:pd.DataFrame, batch_size:int) -> tf.data.Dataset:
    """Returns a batched dataset from an unbatched dataset\n
    Keyword arguments:\n
    df -- dataframe to batch\n
    batch_size -- size of generated batches
    """
    dataset = tf.data.Dataset.from_tensor_slices(df)
    dataset = dataset.batch(batch_size)
    return dataset

def generate_features_and_targets(batch: tf.data.Dataset, min_length:int, max_length:int): 
    """Returns the features and targets for a given batch
    batch: batch to generate features and targets from
    min_length: minimum feature length
    max_length: maximum feature length
    """
    target_pos = random.randint(min_length, max_length-1)
    df = pd.DataFrame(batch)

    features = df.drop(df.columns[target_pos], axis=1)
    targets = df.drop(df.columns[0], axis=1)
    
    return features.to_numpy(), targets.to_numpy(), target_pos

def dump_unused_memory(): 
    """ Dumps unused memory to avoid memory overflow
    """
    gc.collect() 
    keras.backend.clear_session()

##################################################################### Define metrics and memory ######################################################

batch_test_acc_metric = keras.metrics.Accuracy()
epoch_test_acc_metric = keras.metrics.Accuracy()

# Store test loss
batch_test_loss_history = []

# store test accuracy
batch_test_acc_history = []

# make df of right size filled with nan
df_predicted_distributions = pd.DataFrame(np.full([len(df_test_data), vocab_size], np.nan))

##################################################################### Define model ######################################################

# Compile model
model = Encoder(vocab_size=vocab_size, max_input_size=max_input_size, num_heads=num_heads, num_layers=num_layers, d_model=d_model, d_ff=d_ff, drop_rate=drop_rate)

# Dummy input to buid model
dummy_input = df_test_data.iloc[4,:].tolist()
dummy_input = tf.constant([dummy_input])
dummy_input = tf.constant(dummy_input)
model(dummy_input, training=False)
print('Dummy input passed')

# Load the weights
model.load_weights(weights_path)
loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

##################################################################### Predict with the model ######################################################

for iteration in range(1,num_iterations+1):
    print(f"Starting iteration [{iteration}]/[{num_iterations}]")

    # Generate batches
    if iteration == 1:
        df_to_batch = df_test_data
    if iteration > 1: 
        df_to_batch = df_predicted_sequences

    batched_dataset = generate_batches(df_to_batch, batch_size)

    # Define start and end of features
    start = min_length+iteration-1
    end = start+1

    batch_counter = 0 
    values_list = []
    distributions_list = []
    df_predicted_distributions = pd.DataFrame(np.full([len(df_test_data), vocab_size], np.nan))

    for batch in batched_dataset: 
        batch_test_features, batch_test_targets, target_pos = generate_features_and_targets(batch, start, end)
            
        # Forward pass
        test_output = model(batch_test_features, training=False)
        test_loss = loss_function(batch_test_targets, test_output)

        # Convert predictions
        predicted_distribution = test_output[:,target_pos-1,:].numpy()
        predicted_values = np.argmax(test_output[:,target_pos-1,:], axis=-1)
        
        # Add predicted values and distributions to a list 
        values_list = np.append(values_list, predicted_values)
        distributions_list = np.append(distributions_list, predicted_distribution)

        # Compute accuracy of this batch and store
        batch_test_acc_metric.update_state(batch_test_targets[:,target_pos-1], predicted_values)
        batch_test_acc_history.append(batch_test_acc_metric.result().numpy())
        if verbose:
            print(f"\tBatch [{batch_counter+1}/{num_batches}], last test data = {batch_test_features[-1,:]}, to predict = {batch_test_targets[-1]}, Predicted value: {values_list[-1]}")
            print(f"\t    Accuracy of this batch: {100*batch_test_acc_metric.result().numpy():.2f}%")
        batch_test_acc_metric.reset_state()

        # Add loss to list
        batch_test_loss_history.append(test_loss.numpy())

        batch_counter = batch_counter + 1

    epoch_test_acc_metric.update_state(df_to_batch.iloc[:,target_pos], values_list)
    print(f'\tAccuracy of this iteration: {epoch_test_acc_metric.result().numpy()}')

##################################################################### Store all data ######################################################
    # After each iteration, store data
    
    df_predicted_sequences = df_to_batch.iloc[:,:target_pos].copy()
    df_predicted_sequences[f'Prediction_{iteration}'] = values_list
    df_predicted_sequences = pd.concat([df_predicted_sequences, df_to_batch.iloc[:,end:]], axis=1)

    distributions_list = np.reshape(distributions_list, (len(df_test_data),vocab_size))
    df_predicted_distributions = pd.DataFrame(distributions_list)

    directory = Path(f"{store_path}")
    directory.mkdir(exist_ok=True)

    df_predicted_distributions.to_csv(directory/f'Predicted_distributions_position_{min_length}_{dataset}.csv', index=False)
    print(f'Saved predicted distributions')
    df_predicted_sequences.to_csv(directory/f'Predicted_sequences_{dataset}.csv', index=False)
    print('Saved predicted sequences')

    f = h5py.File(directory/f'History_predictions_{dataset}.hdf5', 'a')
    if not "batch_test_loss_history" in f:
        f.create_dataset('batch_test_loss_history', data=batch_test_loss_history)
    if "batch_test_loss_history" in f: 
        del f['batch_test_loss_history']
        f.create_dataset('batch_test_loss_history', data=batch_test_loss_history)
