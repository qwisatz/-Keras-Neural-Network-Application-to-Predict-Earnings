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


#===================================
#create model object
#===================================
# Define the model
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
# MSE FOR SHORT, DEFAULT CHOICE IF NOT INPUTTED
model.compile(loss="mean_squared_error", optimizer="adam")




##==================================
#2 NTRAIN MODEL PARAMETERS
#===================================
# Train the model
model.fit(
    X,
    Y,
    # LOOPS, CHECK EPOCHS TO KNOW WHEN IT BECOMES ACCURATE
    epochs=50,
    # BEST TRAINING IS WHEN DATA IS SHUFFLED
    shuffle=True,
    # REDUCE LOG MESSAGES
    verbose=2
)




##==================================
# LOAD TEST DATA
#===================================
# Load the separate test data set
test_data_df = pd.read_csv("sales_data_test_scaled.csv")

X_test = test_data_df.drop('total_earnings', axis=1).values
Y_test = test_data_df[['total_earnings']].values

##==================================
# TEST TEST DATA
#===================================
test_error_rate = model.evaluate(X_test, Y_test, verbose=0)
print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))


# Save the model to disk
model.save("trained_model.h5")
print("Model saved to disk.")
