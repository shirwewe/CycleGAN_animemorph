import pandas as pd
import shutil
import os

# Load the CSV file
csv_file_path = '/home/shirwee/documents/aps360-project/output.csv'
df = pd.read_csv(csv_file_path)

# Specify the column name that contains file names and the target file name
file_name_column = 'File Name'  # Change this to your actual column name

# Filter rows where the file name matches the target file name

# Define the source directory where files are located
source_directory = '/home/shirwee/documents/aps360-project/dataSetB_60k'

# Define the destination directory where files will be moved
destination_directory = '/home/shirwee/documents/aps360-project/dataSetB_good'

# Make sure the destination directory exists
os.makedirs(destination_directory, exist_ok=True)

# Iterate over the filtered file names and move them to the destination directory
for file_name in df[file_name_column]:
    source_file_path = os.path.join(source_directory, file_name)
    destination_file_path = os.path.join(destination_directory, file_name)
    
    # Check if the source file exists
    if os.path.exists(source_file_path):
        # Move the file
        shutil.move(source_file_path, destination_file_path)
        print(f"Moved {file_name} to {destination_directory}")
    else:
        print(f"File {file_name} not found in {source_directory}")

print("File moving process completed.")
