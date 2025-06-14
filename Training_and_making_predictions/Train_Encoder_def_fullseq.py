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

##################################################################### Read model parameters ######################################################

parser = argparse.ArgumentParser(description='Training the encoder model')
parser.add_argument('-num_heads', type=int, default=1, help='Number of attention heads')
parser.add_argument('-num_layers', type=int, default=1, help='Number of encoder layers')
parser.add_argument('-d_model', type=int, default=32, help='Dimension of the model d_model')
parser.add_argument('-d_ff', type=int, default=512, help='Dimension of the feed-forward layer')
parser.add_argument('-drop_rate', type=float, default=0.1, help='Drop rate of the dropout layers')
parser.add_argument('-num_epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('-min_length', type=int, default=7, help='Minimun sequence length')
parser.add_argument('-max_length', type=int, default=8, help='Maximum sequence length')
parser.add_argument('-batch_size', type=int, default=256, help='Batch_size')
parser.add_argument('-train_path', type=str, default=None, help='Path to train set')
parser.add_argument('-val_path', type=str, default=None, help='Path to validation set')
parser.add_argument('-store_path', type=str, default=None, help='Path to where model will be saved')

verbose = False
args = parser.parse_args()

num_heads = args.num_heads
num_layers = args.num_layers
d_model = args.d_model
d_ff = args.d_ff
drop_rate = args.drop_rate
num_epochs = args.num_epochs
min_length = args.min_length
max_length = args.max_length
batch_size = args.batch_size
train_path = args.train_path
val_path = args.val_path 
store_path = args.store_path

df_train_data = pd.read_csv(train_path)
df_val_data = pd.read_csv(val_path)

# Define hyper parameters 
vocab_size = df_train_data.max().max()+1+int(np.ceil(0.1*df_train_data.max().max())) # Include 10% of extra data to account for differences between train, val, test and signal/background
max_input_size = len(df_train_data.columns)
n_train_events = len(df_train_data)
n_val_events = len(df_val_data)

num_batches = int(np.ceil(n_train_events/batch_size))
num_val_batches = int(np.ceil(n_val_events/batch_size))


#################################################################### Store train outputs #################################################################

# Define directory of needed
directory = Path(f"{store_path}")
directory.mkdir(exist_ok=True)

# Generate txt file to store model parameters
f_params = open(directory/"parameters.txt", "w")
f_params.write(f"""Number of heads = {num_heads} \nNumber of layers = {num_layers} \nd_model = {d_model} \nd_ff = {d_ff} \nNumber of epochs = {num_epochs}
               \nMinimum position = {min_length} \nMaximum position = {max_length} \nbatch size = {batch_size} \nTrain path = {train_path}""")
f_params.close()

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

# Accuracy metric
batch_train_acc_metric = keras.metrics.Accuracy()
batch_val_acc_metric = keras.metrics.Accuracy()
epoch_train_acc_metric = keras.metrics.Accuracy()
epoch_val_acc_metric = keras.metrics.Accuracy()

# Store train and validation loss
batch_train_loss_history = []
batch_val_loss_history = []
epoch_train_loss_history = []
epoch_val_loss_history = []

# Store train and validation accuracy
batch_train_acc_history = []
batch_val_acc_history = []
epoch_train_acc_history = []
epoch_val_acc_history = []

##################################################################### Define model ######################################################

# Define loss, optimizer and callbacks
loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
optimizer = keras.optimizers.Adam()

# Compile model
model = Encoder(vocab_size=vocab_size, max_input_size=max_input_size, num_heads=num_heads, num_layers=num_layers, d_model=d_model, d_ff=d_ff, drop_rate=drop_rate)
model.compile(optimizer=optimizer, loss=loss_function)

##################################################################### Train the model ######################################################

for epoch in range (1, num_epochs+1): 
    print(f"Starting epoch [{epoch}]/[{num_epochs}]")

    # Generate batches
    train_batched_dataset = generate_batches(df_train_data, batch_size)
    val_batched_dataset = generate_batches(df_val_data, batch_size)

    batch_counter = 0
    values_list = []
    for batch in train_batched_dataset:
        with tf.GradientTape() as tape: 
            # Generate features and targets for a new random position
            batch_train_features, batch_train_targets, target_pos = generate_features_and_targets(batch, min_length, max_length)
            
            # Forward pass
            train_output = model(batch_train_features, training=True)
            total_train_loss = loss_function(batch_train_targets, train_output) # loss is already averaged over batch size
            # Add batch train loss to list
            batch_train_loss_history.append(total_train_loss.numpy())

            # Convert predictions
            train_predicted_values = np.argmax(train_output[:,target_pos-1,:], axis=-1)

            # Add predicted values to a list 
            values_list = np.append(values_list, train_predicted_values)

            # Compute accuracy of this batch and store
            batch_train_acc_metric.update_state(batch_train_targets[:,target_pos-1], train_predicted_values)
            batch_train_acc_history.append(batch_train_acc_metric.result().numpy())
            if verbose:
                print(f"\t\tBatch [{batch_counter+1}/{num_batches+1}], last train data = {batch_train_features[-1,:]}, Target = {batch_train_targets[-1,target_pos-1]}")
                print(f"\t\t    Accuracy of this batch: {100*batch_train_acc_metric.result().numpy():.2f}%")
                print(f"\t\t    Train loss of this batch: {total_train_loss.numpy()}")
            batch_train_acc_metric.reset_state()

            batch_counter = batch_counter + 1
        
        # Backwards pass after each batch 
        gradients = tape.gradient(total_train_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # Compute train accuracy over this epoch
    epoch_train_acc_metric.update_state(df_train_data.iloc[:,target_pos], values_list)
    epoch_train_acc_history.append(epoch_train_acc_metric.result().numpy())
    print(f"\tTrain accuracy of this epoch: {100*epoch_train_acc_metric.result().numpy():.2f}%")
    epoch_train_acc_metric.reset_state()

    # Add epoch train loss to list
    epoch_train_loss_history.append(np.mean(batch_train_loss_history[-int(num_batches):])) 
    print(f"\tTrain loss of this epoch: {epoch_train_loss_history[-1]}")

    # Run validation data after each epoch 
    print("\tStarting validation set")
    val_batch_counter = 0
    val_values_list = []
    for val_batch in val_batched_dataset:
        val_batch_features, val_batch_targets, val_target_pos = generate_features_and_targets(val_batch, min_length, max_length)

        # Forward pass
        val_output = model(val_batch_features, training=False)
        total_val_loss = loss_function(val_batch_targets, val_output)
        if verbose:
            print(f'\t    1. val_loss: {total_val_loss}')
            print(f'\t    2. batch length: {len(val_batch[:,0])}')

        # Add loss to list
        batch_val_loss_history.append(total_val_loss.numpy())

        # Convert predictions
        val_predicted_values = np.argmax(val_output[:,val_target_pos-1,:], axis=-1)

        # Add predicted values to a list 
        val_values_list = np.append(val_values_list, val_predicted_values)

        # Compute accuracy of this batch and store
        batch_val_acc_metric.update_state(val_batch_targets[:,val_target_pos-1], val_predicted_values)
        batch_val_acc_history.append(batch_val_acc_metric.result().numpy())
        if verbose:
            print(f"\t    Batch [{val_batch_counter+1}/{num_val_batches+1}], last validation data = {val_batch_features[-1,:]}")
            print(f"\t    Accuracy of this batch: {100*batch_val_acc_metric.result().numpy():.2f}%")
        batch_val_acc_metric.reset_state()

        val_batch_counter = val_batch_counter + 1 

    # Compute validation accuracy over this epoch
    epoch_val_acc_metric.update_state(df_val_data.iloc[:,val_target_pos], val_values_list)
    epoch_val_acc_history.append(epoch_val_acc_metric.result().numpy())
    print(f"\tValidation accuracy of this epoch: {100*epoch_val_acc_metric.result().numpy():.2f}%")
    epoch_val_acc_metric.reset_state()

    # Add epoch val loss to list
    epoch_val_loss_history.append(np.mean(batch_val_loss_history[-int(num_val_batches):]))
    if verbose:
        print(f'\t    3. epoch_val_loss_history: {np.mean(batch_val_loss_history[-int(num_val_batches):])}')
        print(f'\t    4. epoch_val_loss_history: {batch_val_loss_history[-int(num_val_batches):]}')
        print(f'\t    5. epoch_val_loss_history: {batch_val_loss_history, len(batch_val_loss_history)}')
    print(f"\tValidation loss of this epoch: {epoch_val_loss_history[-1]}")

    # Shuffle data to get variation in data within batches
    df_train_data = df_train_data.sample(frac=1)
    df_val_data = df_val_data.sample(frac=1)

    dump_unused_memory()

##################################################################### Store all data ######################################################

    # After each epoch, store data

    f = h5py.File(directory/'History_notebook.hdf5', 'a')
    model.save(directory/'Model_notebook.keras')
    model.save_weights(directory/"Model_notebook.weights.h5")

    f.attrs["vocab_size"] = vocab_size
    f.attrs["max_input_size"] = max_input_size
    f.attrs["drop_rate"] = drop_rate
    f.attrs["d_model"] = d_model
    f.attrs["d_ff"] = d_ff
    f.attrs["h"] = num_heads
    f.attrs["n"] = num_layers
    f.attrs["num_epochs"] = num_epochs
    f.attrs["target_pos"] = target_pos
    f.attrs["batch_size"] = batch_size
    f.attrs["train_path"] = train_path

    if not "batch_train_loss_history" in f:
        f.create_dataset('batch_train_loss_history', data=batch_train_loss_history)
    if "batch_train_loss_history" in f: 
        del f['batch_train_loss_history']
        f.create_dataset('batch_train_loss_history', data=batch_train_loss_history)
    
    if not "batch_val_loss_history" in f:
        f.create_dataset('batch_val_loss_history', data=batch_val_loss_history)
    if "batch_val_loss_history" in f: 
        del f['batch_val_loss_history']
        f.create_dataset('batch_val_loss_history', data=batch_val_loss_history)

    if not "epoch_train_loss_history" in f:
        f.create_dataset('epoch_train_loss_history', data=epoch_train_loss_history)
    if "epoch_train_loss_history" in f: 
        del f['epoch_train_loss_history']
        f.create_dataset('epoch_train_loss_history', data=epoch_train_loss_history)
    
    if not "epoch_val_loss_history" in f:
        f.create_dataset('epoch_val_loss_history', data=epoch_val_loss_history)
    if "epoch_val_loss_history" in f: 
        del f['epoch_val_loss_history']
        f.create_dataset('epoch_val_loss_history', data=epoch_val_loss_history)  
    
    if not "batch_train_acc_history" in f: 
        f.create_dataset('batch_train_acc_history', data=batch_train_acc_history)
    if "batch_train_acc_history" in f: 
        del f['batch_train_acc_history']
        f.create_dataset('batch_train_acc_history', data=batch_train_acc_history)

    if not "batch_val_acc_history" in f: 
        f.create_dataset('batch_val_acc_history', data=batch_val_acc_history)
    if "batch_val_acc_history" in f: 
        del f['batch_val_acc_history']
        f.create_dataset('batch_val_acc_history', data=batch_val_acc_history)

    if not "epoch_train_acc_history" in f: 
        f.create_dataset('epoch_train_acc_history', data=epoch_train_acc_history)
    if "epoch_train_acc_history" in f: 
        del f['epoch_train_acc_history']
        f.create_dataset('epoch_train_acc_history', data=epoch_train_acc_history)

    if not "epoch_val_acc_history" in f: 
        f.create_dataset('epoch_val_acc_history', data=epoch_val_acc_history)
    if "epoch_val_acc_history" in f: 
        del f['epoch_val_acc_history']
        f.create_dataset('epoch_val_acc_history', data=epoch_val_acc_history)

    f.close()  