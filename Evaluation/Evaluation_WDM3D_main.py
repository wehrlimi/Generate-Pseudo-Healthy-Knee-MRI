import pdb
import nibabel as nib
import torch
import numpy as np
import os
import pandas as pd
from Evaluation_utils import compute_metrics
import datetime

# Define the paths to the folders
gt_folder_path = r'C:\Users\michael\Documents\Data\Fresh\fastMRI_nifti\test_3D_gt'
sample_folder_path = r'C:\Users\michael\Desktop\WDM_3D_Sampling_Results\fastmri\sampling_output'
mask_folder_path = r'C:\Users\michael\Documents\Data\Fresh\fastMRI_nifti\test_3D'  # Path to the masks folder

def load_nii_to_tensor(file_path):
    """Load a NIfTI file and convert it to a PyTorch tensor."""
    nii_img = nib.load(file_path)
    data = nii_img.get_fdata()
    tensor = torch.tensor(data, dtype=torch.float32)
    return tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

def find_files_in_subdirs(folder_path, part, keyword):
    """Recursively find all files in the subdirectories that match a keyword and return a map of identifiers to file paths."""
    file_map = {}
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            parts = file_name.split('_')
            #pdb.set_trace()
            if len(parts) > part and keyword in file_name:
                identifier = parts[part]
                file_path = os.path.join(root, file_name)
                file_map[identifier] = file_path
    return file_map

# Create mappings for files based on specific keywords
sample_folder_path_map = find_files_in_subdirs(sample_folder_path, 3, 'sample')  # Filtering for 'sample'
gt_folder_path_map = find_files_in_subdirs(gt_folder_path, 0, 'healthy')  # Filtering for 'healthy'
mask_map = find_files_in_subdirs(mask_folder_path, 0, 'mask')  # Filtering for 'mask'

# Debugging with pdb to inspect variables if needed
#pdb.set_trace()

# List to store the results
results = []

# Additional processing code goes here (e.g., computing metrics, analyzing the data)


# Iterate over each file in the sample folder
for identifier, file_path_2 in sample_folder_path_map.items():
    if identifier in gt_folder_path_map and identifier in mask_map:
        file_path_1 = gt_folder_path_map[identifier]
        mask_file = mask_map[identifier]

        # Load the images and the mask
        image1 = load_nii_to_tensor(file_path_1)
        image2 = load_nii_to_tensor(file_path_2)
        mask = load_nii_to_tensor(mask_file)

        # Check if the mask is empty
        if mask.sum().item() == 0:
            # Append the results with an indication of an empty mask
            results.append({
                'Ground Truth': file_path_1,
                'Mask': mask_file,
                'Sample Image': file_path_2,
                'MSE': 'empty mask',
                'PSNR': 'empty mask',
                'PSNR_01': 'empty mask',
                'SSIM': 'empty mask'
            })
            continue  # Skip to the next example


        # Compute the metrics
        mse, psnr, psnr_01, ssim = compute_metrics(image1, image2, mask)

        # Append the results to the list
        results.append({
            'Ground Truth': file_path_1,
            'Mask': mask_file,
            'Sample Image': file_path_2,
            'MSE': mse,
            'PSNR': psnr,
            'PSNR_01': psnr_01,
            'SSIM': ssim
        })

# Convert the results list to a DataFrame and save as a CSV
results_df = pd.DataFrame(results)

# Calculate the total number of valid samples (exclude rows with 'empty mask')
valid_samples_df = results_df[results_df['MSE'] != 'empty mask']

# Convert the metric columns to numeric to calculate mean and std
valid_samples_df[['MSE', 'PSNR', 'PSNR_01', 'SSIM']] = valid_samples_df[['MSE', 'PSNR', 'PSNR_01', 'SSIM']].apply(pd.to_numeric)

# Calculate the mean and standard deviation for each metric
mean_values = valid_samples_df[['MSE', 'PSNR', 'PSNR_01', 'SSIM']].mean()
std_values = valid_samples_df[['MSE', 'PSNR', 'PSNR_01', 'SSIM']].std()

# Create a summary row
summary_row = {
    'Ground Truth': 'Summary',
    'Mask': '',
    'Sample Image': '',
    'MSE': f"Mean: {mean_values['MSE']:.4f}, Std: {std_values['MSE']:.4f}",
    'PSNR': f"Mean: {mean_values['PSNR']:.4f}, Std: {std_values['PSNR']:.4f}",
    'PSNR_01': f"Mean: {mean_values['PSNR_01']:.4f}, Std: {std_values['PSNR_01']:.4f}",
    'SSIM': f"Mean: {mean_values['SSIM']:.4f}, Std: {std_values['SSIM']:.4f}"
}

# Convert the summary_row into a DataFrame
summary_df = pd.DataFrame([summary_row])

# Append the summary row to the DataFrame using pd.concat
results_df = pd.concat([results_df, summary_df], ignore_index=True)

# Extract the last part of the sample folder path
name = os.path.basename(sample_folder_path)

# Get the current date and time
date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# Save the DataFrame to a CSV file with the name included
results_df.to_csv(fr'C:\Users\michael\Desktop\WDM_3D_Sampling_Results\fastmri\evaluation_results_{name}_{date}.csv', index=False)

print("Evaluation complete. Results saved.")