"""
Loads Route Generation Model
(RNN)
"""

import sys
import numpy as np
from collections import Counter
import pickle
import tensorflow as tf
import keras.backend as K
from keras.models import  Model
from keras.layers import Dense, Activation, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.utils import to_categorical
from keras import backend as K

rnn_dir = 'therock/models/rnn/'
sys.path.append('therock/')
sys.path.append(rnn_dir)
from DeepRouteSetHelper import *



# setterList = []
# countNumOfErrorUsername = 0
# for key in MoonBoard_2016_withurl.keys(): 
#     try:
#         setterList.append(MoonBoard_2016_withurl[key]['setter']['Nickname'])
#     except:
#         countNumOfErrorUsername += 1

# setterDict = {k: v for k, v in sorted(Counter(setterList).items(), key=lambda item: item[1], reverse = True)}

# # add setter with 50+ experience and Benchmark setter
# goodSetterName = []
# for key in setterDict.keys():
#     if setterDict[key] > 50:
#         goodSetterName.append(key)
# for key in MoonBoard_2016_withurl.keys(): 
#     try:
#         if MoonBoard_2016_withurl[key]['isBenchmark'] == True:
#             goodSetterName.append(MoonBoard_2016_withurl[key]['setter']['Nickname'])
#     except:
#         pass

# count = 0
# goodProblemKeyList = []
# for key in MoonBoard_2016_withurl.keys():  
#     try:
#         if MoonBoard_2016_withurl[key]['isBenchmark'] == True:
#             goodSetterName.append(MoonBoard_2016_withurl[key]['setter']['Nickname'])   
#     except:
#         pass

# for key in MoonBoard_2016_withurl.keys():  
#     try:
#         if MoonBoard_2016_withurl[key]['setter']['Nickname'] in goodSetterName:
#             goodProblemKeyList.append(key)
#             count = count + 1
#         if MoonBoard_2016_withurl[key]['isBenchmark'] == True:
#             goodProblemKeyList.append(key)
#             count = count + 1
#         if MoonBoard_2016_withurl[key]['repeats'] > 50:
#             goodProblemKeyList.append(key)
#             count = count + 1
#         if MoonBoard_2016_withurl[key]['num_stars'] == 3:
#             goodProblemKeyList.append(key) 
#             count = count + 1
#     except:
#         pass
# print ("Total amount of good problems: ", count)


# easyProblemKeyList = []
# mediumProblemKeyList = []
# hardProblemKeyList = []
# for key in goodProblemKeyList:
#     if MoonBoard_2016_withurl[key]['grade'] in ["6B+", "6C", "6C+"]: # V4 V5
#             easyProblemKeyList.append(key)
#     if MoonBoard_2016_withurl[key]['grade'] in ["7A", "7A+", "7B", "7B+"]: # V6 7 8
#             mediumProblemKeyList.append(key)
#     if MoonBoard_2016_withurl[key]['grade'] in ["7B", "7B+", "7C", "7C+", "8A", "8A+", "8B"]: # V8 9 10 11 12 13
#             hardProblemKeyList.append(key)        

handStringList = []
for key in benchmark_handString_seq.keys():
    handStringList.append(benchmark_handString_seq[key])

# handStringList = collectHandStringIntoList(mediumProblemKeyList)
# numOfTrainingSample = len(handStringList)

# with open(parent_wd + "/raw_data/holdStr_to_holdIx", 'rb') as f:
#     holdStr_to_holdIx = pickle.load(f)
with open(parent_wd + "/raw_data/holdIx_to_holdStr", 'rb') as f:
    holdIx_to_holdStr = pickle.load(f)  
# numOfPossibleHolds = 277    

# X, Y, n_values = loadSeqXYFromString(handStringList, holdStr_to_holdIx, m = numOfTrainingSample, maxNumOfHands = 12, numOfPossibleHolds = numOfPossibleHolds)
# print(f'n_values: {n_values}')

n_values = 278

n_a = 64 
reshapor = Reshape((1, n_values))                  
LSTM_cell = LSTM(n_a, return_state = True)        
densor = Dense(n_values, activation='softmax')

def deepRouteSet(LSTM_cell, densor, n_values=n_values, n_a=64, Ty=12):
    """
    Uses the trained "LSTM_cell" and "densor" from model() to generate a sequence of values.
    
    Arguments:
    LSTM_cell -- the trained "LSTM_cell" from model(), Keras layer object
    densor -- the trained "densor" from model(), Keras layer object
    n_values -- integer, number of unique values
    n_a -- number of units in the LSTM_cell
    Ty -- integer, number of time steps to generate
    
    Returns:
    inference_model -- Keras model instance
    """
    def one_hot(x):
        x = K.argmax(x)
        x = tf.one_hot(x, n_values) 
        x = RepeatVector(1)(x)
        return x
    
    # Define the input of your model with a shape 
    x0 = Input(shape=(1, n_values))
    
    # Define s0, initial hidden state for the decoder LSTM
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    x = x0

    ### START CODE HERE ###
    # Step 1: Create an empty list of "outputs" to later store your predicted values (≈1 line)
    outputs = []
    
    # Step 2: Loop over Ty and generate a value at every time step
    for t in range(Ty):
        
        # Step 2.A: Perform one step of LSTM_cell (≈1 line)
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        
        # Step 2.B: Apply Dense layer to the hidden state output of the LSTM_cell (≈1 line)
        out = densor(a)

        # Step 2.C: Append the prediction "out" to "outputs". out.shape = (None, n_values) (≈1 line)
        outputs.append(out)
        
        # Step 2.D: 
        # Select the next value according to "out",
        # Set "x" to be the one-hot representation of the selected value
        # See instructions above.
        x = Lambda(one_hot)(out)
        
        
    # Step 3: Create model instance with the correct "inputs" and "outputs" (≈1 line)
    inference_model = Model(inputs = [x0, a0, c0], outputs = outputs)
    
    ### END CODE HERE ###
    return inference_model


def predict_and_sample(inference_model, 
                       x_initializer=np.random.rand(1, 1, n_values) / 100, 
                       a_initializer=np.random.rand(1, n_a) * 150, 
                       c_initializer=np.random.rand(1, n_a) / 2):
    """
    Predicts the next value of values using the inference model.
    
    Arguments:
    inference_model -- Keras model instance for inference time
    x_initializer -- numpy array of shape (1, 1, n_values), one-hot vector initializing the values generation
    a_initializer -- numpy array of shape (1, n_a), initializing the hidden state of the LSTM_cell
    c_initializer -- numpy array of shape (1, n_a), initializing the cell state of the LSTM_cel
    
    Returns:
    results -- numpy-array of shape (Ty, n_values), matrix of one-hot vectors representing the values generated
    indices -- numpy-array of shape (Ty, 1), matrix of indices representing the values generated
    """
    
    # Step 1: Use your inference model to predict an output sequence given x_initializer, a_initializer and c_initializer.
    pred = inference_model.predict([x_initializer, a_initializer, c_initializer])
    # Step 2: Convert "pred" into an np.array() of indices with the maximum probabilities
    indices =  np.argmax(pred, axis = 2)
    # Step 3: Convert indices to one-hot vectors, the shape of the results should be (Ty, n_values)
    results =  to_categorical(indices, num_classes = np.shape(x_initializer)[2])
    
    return results, indices

inference_model = deepRouteSet(LSTM_cell, densor, n_values=n_values, n_a=64, Ty=12)
inference_model.load_weights(rnn_dir + "DeepRouteSetMedium_v1.h5")
