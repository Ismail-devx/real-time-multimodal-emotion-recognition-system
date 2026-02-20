import zipfile
import os

# dataset names and paths
dataset_names = ['ravdess', 'savee', 'crema_d', 'goemotions', 'dailydialog']
base_path = './Datasets/'

# Function to unzip dataset if not already extracted
def unzip_dataset(dataset_name):
    zip_path = os.path.join(base_path, f'{dataset_name}.zip')
    extract_path = os.path.join(base_path, dataset_name)

    # Check if the zip file exists
    if not os.path.exists(zip_path):
        print(f" {zip_path} not found. ")
        return

    # Check if already extracted
    if not os.path.exists(extract_path):
        os.makedirs(extract_path, exist_ok=True)
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            print(f" Successfully extracted {dataset_name} to {extract_path}")
        except zipfile.BadZipFile:
            print(f" error: {dataset_name}.zip is not a valid zip file.")
    else:
        print(f" {dataset_name} is already extracted at {extract_path}")

# Loop through and unzip all datasets
for dataset in dataset_names:
    unzip_dataset(dataset)
