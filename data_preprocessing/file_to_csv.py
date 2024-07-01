import pandas as pd
import shutil
import os
from PIL import Image

def resize_images(folder, size=(128, 128)):
    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)
        if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            with Image.open(file_path) as img:
                img = img.resize(size, Image.Resampling.LANCZOS)
                img.save(file_path)

# Example usage:
folder = r'/home/shirwee/documents/aps360-project/dataSetB_60k/images'
resize_images(folder)

# def create_folder(destination_folder):
#     os.makedirs(destination_folder, exist_ok=True)
#     return destination_folder

# def copy_files(file_list, source_folder, destination_folder):
#     for file_name in file_list:
#         src_file = os.path.join(source_folder, file_name)
#         dest_file = os.path.join(destination_folder, file_name)
#         shutil.copy2(src_file, dest_file)

# def pick_and_copy_photos(csv_file, source_folder, destination_folder, num_photos=5000):
#     # Read the CSV file into a DataFrame
#     df = pd.read_csv(csv_file)
    
#     # Check if the required columns are present
#     if 'image_id' not in df.columns or 'Male' not in df.columns:
#         raise ValueError("CSV file must contain 'image_id' and 'Male' columns")
    
#     # Filter out the photos by gender and select the first num_photos (5000) entries
#     male_photos = df[df['Male'] == 1]['image_id'].tolist()[:num_photos]
#     female_photos = df[df['Male'] == -1]['image_id'].tolist()[:num_photos]
    
#     # Ensure there are enough photos
#     if len(male_photos) < num_photos or len(female_photos) < num_photos:
#         raise ValueError(f"Not enough photos to select {num_photos} male and female photos")
    
#     # Create the destination folder
#     destination_folder = create_folder(destination_folder)
    
#     # Copy the selected photos
#     copy_files(male_photos, source_folder, destination_folder)
#     copy_files(female_photos, source_folder, destination_folder)

# # Example usage:
# csv_file = r'C:\Users\Shirley Li\Documents\2024-2025\aps360\aps360-project\dataSetA_200k\list_attr_celeba.csv'
# source_folder = r'C:\Users\Shirley Li\Documents\2024-2025\aps360\aps360-project\dataSetA_200k\img_align_celeba\img_align_celeba'
# destination_folder = r'C:\Users\Shirley Li\Documents\2024-2025\aps360\aps360-project\selected_photos'

# pick_and_copy_photos(csv_file, source_folder, destination_folder)