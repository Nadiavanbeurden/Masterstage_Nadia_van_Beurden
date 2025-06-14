# Masterstage Nadia van Beurden
The encoder to predict sequences of particle detector data.

## Data
These python scripts assume a dataformat as described in my thesis.

## Guide
1. Tokenizing and spliting your dataset<br/>
   If you want to tokenize before splitting into a signal and background dataset, use: `generate_dataset_full.py`. If you first want to split between signal and background before tokenizing, choose `generate_dataset.py`. For spliting the background and signal datasets into train, test and validation sets, you can use split.py These .py files can all be found in the folder: `Tokenizing_and_splitting_dataset`.
3. Training phase <br/>
   Use the folder `Training_and_making_predictions` and the file `Train_Encoder_def_fullseq.py`
5. Predicting phase <br/>
   Use the folder `Training_and_making_predictions` and the file `Predict_Encoder_def_fullseq.py`
