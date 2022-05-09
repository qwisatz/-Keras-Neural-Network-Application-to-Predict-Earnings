import pandas as pd
from keras.models import Sequential
from keras.layers import *

training_data_df = pd.read_csv("sales_data_training_scaled.csv")

X = training_data_df.drop('total_earnings', axis=1).values
Y = training_data_df[['total_earnings']].values

# Define the model
model = Sequential()
model.add(Dense(50, input_dim=9, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(
    X,
    Y,
    epochs=50,
    shuffle=True,
    verbose=2
)

# Load the separate test data set
test_data_df = pd.read_csv("sales_data_test_scaled.csv")

X_test = test_data_df.drop('total_earnings', axis=1).values
Y_test = test_data_df[['total_earnings']].values

test_error_rate = model.evaluate(X_test, Y_test, verbose=0)
print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))






##==================================
# LOAD NEW CSV DATA TO MAKE A PREDICTION
#===================================
# Load the TEST data we make to use to make a prediction
# MISSING TOTAL EARNINGS
X = pd.read_csv("proposed_new_product.csv").values

# Make a prediction with the neural network
# X IS INPUT 
prediction = model.predict(X)

# Grab just the first element of the first prediction (since that's the only have one)
# WILL BE IN 0-1 RANGE BECAUSE SCALED DATA
# 2 D ARRAY
# KERAS ALWAYS ASSUMES THAT WE ARE GOING TO ASK FOR
# MULTIPLE PREDICTIONS WITH MULTIPLE OUTPUT VALUES
# SO IT ALWAYS RETURS PREDICTIONS IN A 2D ARRAY
# WE JUST CARE FIRST VALUE FOR FIRST PREDICTION
# GRAB ELEMENT [0] [0]
prediction = prediction[0][0]

# Re-scale the data from the 0-to-1 range back to dollars
# These constants are from when the data was originally scaled down to the 0-to-1 range
# MUST HAVE YOUR SCALER VALUES TO REVERSE
# IT WAS
# Note: total_earnings values were
# scaled by multiplying by 0.0000036968 and adding -0.115913
prediction = prediction + 0.1159
prediction = prediction / 0.0000036968

print("Earnings Prediction for Proposed Product - ${}".format(prediction))
