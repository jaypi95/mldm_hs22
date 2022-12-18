import pandas as pd
from sklearn.ensemble import RandomForestClassifier

##### LOADING AND PREPROCESSING TRAINING DATA ######
print("Loading training data")
train_data = pd.read_csv("train.csv")

# Convert the Dates to a numerical format
print("Preprocessing training data")
train_data["Year"] = pd.to_datetime(train_data["Dates"]).dt.year
train_data["Month"] = pd.to_datetime(train_data["Dates"]).dt.month
train_data["Day"] = pd.to_datetime(train_data["Dates"]).dt.day

# Convert the dayofweek and pdistrict columns using one-hot encoding so we have them as numbers
train_data = pd.get_dummies(train_data, columns=["DayOfWeek"], prefix="", prefix_sep="", dtype=int, dummy_na=False)
train_data = pd.get_dummies(train_data, columns=["PdDistrict"], prefix="", prefix_sep="", dtype=int, dummy_na=False)

# Drop the Address column
train_data = train_data.drop("Address", axis=1)

# Split the training data into features and target
X_train = train_data[["Year", "Month", "Day", "X", "Y"] + [col for col in train_data.columns if col.startswith("DayOfWeek_")] + [col for col in train_data.columns if col.startswith("PdDistrict_")]]
y_train = train_data["Category"]

##### END OF LOADING AND PREPROCESSING TRAINING DATA ######

##### LOADING AND PREPROCESSING TEST DATA ######

print("Loading test data")
test_data = pd.read_csv("test.csv")

# Convert the Dates to a numerical format
print("Preprocessing test data")
test_data["Year"] = pd.to_datetime(test_data["Dates"]).dt.year
test_data["Month"] = pd.to_datetime(test_data["Dates"]).dt.month
test_data["Day"] = pd.to_datetime(test_data["Dates"]).dt.day

# Convert the dayofweek and pdistrict columns using one-hot encoding so we have them as numbers
test_data = pd.get_dummies(test_data, columns=["DayOfWeek"], prefix="", prefix_sep="", dtype=int, dummy_na=False)
test_data = pd.get_dummies(test_data, columns=["PdDistrict"], prefix="", prefix_sep="", dtype=int, dummy_na=False)

# Drop the Address column
test_data = test_data.drop("Address", axis=1)

# Split the test data into features and target
X_test = test_data[["Year", "Month", "Day", "X", "Y"] + [col for col in test_data.columns if col.startswith("DayOfWeek_")] + [col for col in test_data.columns if col.startswith("PdDistrict_")]]

##### END OF LOADING AND PREPROCESSING TEST DATA ######

##### TRAINING AND PREDICTING ######

# Train the model
model = RandomForestClassifier()
print(f"Training the model by using {model}")
model.fit(X_train, y_train)

# Make predictions on the test set
print("Make predictions")
y_pred = model.predict(X_test)

##### END OF TRAINING AND PREDICTING ######

##### POSTPROCESSING DATA ######

# Create a DataFrame with the predictions and the Id column
print("Process the final result data")
results = pd.DataFrame({"Id": test_data["Id"], "Category": y_pred})

# Convert the Category column to binary columns using the categories from the training data
results = pd.get_dummies(results, columns=["Category"], prefix="", prefix_sep="", dtype=int, dummy_na=False)

# For some reason TREA never gets predicted but it is required so we're adding it here manually
results["TREA"] = 0


# Save the results to a CSV file
results.to_csv("results.csv", index=False)

##### END OF POSTPROCESSING DATA ######