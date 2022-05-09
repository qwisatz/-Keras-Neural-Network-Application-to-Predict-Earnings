import pandas as pd
from sklearn.preprocessing import MinMaxScaler
#===================================
# TRY TO PREDICT TOTOAL EARINGS USING ALL THE OTHER COLOUMS
#===================================


# Load training data set from CSV file
training_data_df = pd.read_csv("sales_data_training.csv")

# Load testing data set from CSV file
test_data_df = pd.read_csv("sales_data_test.csv")

#===================================
#SCALE DATA
#===================================
# Data needs to be scaled to a small range like 0 to 1 for the neural
# network to work well.
# SCLAING ALL DAT FROM 0-1 INSTEAD OF 43 -12414
scaler = MinMaxScaler(feature_range=(0, 1))


##==================================
# IMPORT DATA TO SACALING FUNCTION
#===================================
# Scale both the training inputs and outputs
scaled_training = scaler.fit_transform(training_data_df)
scaled_testing = scaler.transform(test_data_df)


#===================================
# FIND SCALER NUMBER
#===================================
# Print out the adjustment that the scaler applied to the total_earnings column of data
print("Note: total_earnings values were scaled by multiplying by {:.10f} and adding {:.6f}".format(scaler.scale_[8], scaler.min_[8]))


#===================================
# SCALE THE OG DATA WITH THE SCALER
#===================================
# Create new pandas DataFrame objects from the scaled data
scaled_training_df = pd.DataFrame(scaled_training, columns=training_data_df.columns.values)
scaled_testing_df = pd.DataFrame(scaled_testing, columns=test_data_df.columns.values)



#===================================
# SAVE THE SCALED DATA IN A NEW CSV
#===================================
# Save scaled data dataframes to new CSV files
scaled_training_df.to_csv("sales_data_training_scaled.csv", index=False)
scaled_testing_df.to_csv("sales_data_testing_scaled.csv", index=False)
