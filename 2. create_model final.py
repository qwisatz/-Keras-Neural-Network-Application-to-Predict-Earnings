import pandas as pd
from keras.models import Sequential
from keras.layers import *


#===================================
#LOADING PRESCALED DATA FROM OTHER PYTHON FILE
#===================================
# THIS DATA WAS SCALED USING PANDA FROM OTHER FILE
# FROM FILE PRE PROCESS DATA PYTHON
training_data_df = pd.read_csv("sales_data_training_scaled.csv")

X = training_data_df.drop('total_earnings', axis=1).values
Y = training_data_df[['total_earnings']].values

# Define the model
#===================================
#create model object
#===================================
model = Sequential()

#===================================
#ADD LAYERS
#===================================

# densly connected layer is WHERE EVERY NODE IS CONNECTED TO EVERY NODE
# ACTIVATION FUNCTION 'RELU' TO START NERUAL NETWORK
# DEFAULT SETTING ARE THERE, JSUT CHOOSE NUMBER OF LAYERS AND ACTIVATION FN
# INPUT = 9 BECAUS E 9 COLOUMNS IN CSV
model.add(Dense(50, input_dim=9, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
# LAST LAYER
# PREDICTED VALUES SO BE 1 OUTPUT LINEAR VALUE
# THIS IS DEFAULT SO YOU DON'T HAVE TO TYPE IT IF YOU WANT
model.add(Dense(1, activation='linear'))

#===================================
# MEASURE ACCRUACY AFTER EVERY LOOP IS CALLED LOSS FUNCTION
# ALSO USE OPTIMERS
#===================================
# OPTIMISER IS HOW TO TRAIN YOUR NEURAL NETWORK
# MSE FOR SHORT, DEAFULT GOOD CHOICE
model.compile(loss="mean_squared_error", optimizer="adam")
