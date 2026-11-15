import pandas as pd

# save filepath to variable for easier access
melbourne_file_path = "E:\\ANK\\kaggle\\melb_data.csv"
# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path) 
# print a summary of the data in Melbourne data
melbourne_data.describe()
print(melbourne_data.describe())

iowa_file_path = "E:\\ANK\\kaggle\\train.csv"
# read the data and store data in DataFrame titled melbourne_data
iowa_data = pd.read_csv(melbourne_file_path) 
# print a summary of the data in Melbourne data
iowa_data.describe()
print(iowa_data.describe())
