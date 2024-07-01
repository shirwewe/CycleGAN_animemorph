import os
import csv
from datetime import datetime

# Define the input folder and the output CSV file path
input_folder = '/home/shirwee/documents/aps360-project/dataSetB_good_landmark'
output_csv = 'output.csv'

# Gather file information
file_data = []
for file_name in os.listdir(input_folder):
    file_path = os.path.join(input_folder, file_name)
    if os.path.isfile(file_path):  # Check if it's a file
        file_info = os.stat(file_path)
        file_data.append({
            'File Name': file_name,
            'File Size (Bytes)': file_info.st_size,
            'Creation Date': datetime.fromtimestamp(file_info.st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
            'Last Modified Date': datetime.fromtimestamp(file_info.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        })

# Write data to CSV
with open(output_csv, 'w', newline='') as csvfile:
    fieldnames = ['File Name', 'File Size (Bytes)', 'Creation Date', 'Last Modified Date']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for row in file_data:
        writer.writerow(row)

print(f"File information saved to {output_csv}")
