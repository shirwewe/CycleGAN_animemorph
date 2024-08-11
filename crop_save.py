import os
from PIL import Image

def crop_and_save_images(input_folder, output_folder, crop_size=(130, 132)):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):  # Adjust extensions as needed
            file_path = os.path.join(input_folder, filename)

            # Open the image
            with Image.open(file_path) as img:
                # Calculate coordinates for cropping the center of the image
                width, height = img.size
                left = (width - crop_size[0]) / 2
                top = (height - crop_size[1]) / 2
                right = (width + crop_size[0]) / 2
                bottom = (height + crop_size[1]) / 2

                # Crop the image
                cropped_img = img.crop((left, top, right, bottom))

                # Save the cropped image to the output folder
                output_path = os.path.join(output_folder, filename)
                cropped_img.save(output_path)
                print(f"Cropped and saved {filename} to {output_path}")

# Set the input and output folder paths
input_folder = "/home/shirwee/documents/aps360-project/more_output/outputwnoise/htoa"  # Replace with the path to your input folder
output_folder = "/home/shirwee/documents/aps360-project/more_output/outputwnoise/htoa_cropped"  # Replace with the path to your output folder

# Call the function to crop and save images
crop_and_save_images(input_folder, output_folder)
