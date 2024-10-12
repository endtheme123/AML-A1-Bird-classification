import os
import pandas as pd
import shutil

# Paths
image_folder = '/home/endtheme/git/bird-class/data/Train'
annotation_file = '/home/endtheme/git/bird-class/data/train.txt'
new_image_folder = '/home/endtheme/git/bird-class/data/TrainCompact'
new_annotation_file = '/home/endtheme/git/bird-class/data/compact.txt'

# Create new folder if it doesn't exist
os.makedirs(new_image_folder, exist_ok=True)

# Read annotation file

root = "/home/endtheme/git/bird-class/data/"
txt_path = os.path.join(root, "train.txt")

annotations  = pd.read_csv(txt_path, sep=" ", names= ['image_name', "label"])
# Group by label and select one image per class
selected_images = annotations.groupby('label').first().reset_index()

# Copy selected images to new folder
for _, row in selected_images.iterrows():
    img_name = row['image_name']
    label = row['label']
    
    # Copy the image
    src = os.path.join(image_folder, img_name)
    dst = os.path.join(new_image_folder, img_name)
    shutil.copyfile(src, dst)

# Save the new annotation file
selected_images.to_csv(new_annotation_file, index=False)

print("Process complete! Images and new annotation file are saved.")