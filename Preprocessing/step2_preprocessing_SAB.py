import os
import pandas as pd
import nibabel as nib
from scipy.ndimage import zoom, median_filter, gaussian_filter, binary_erosion, binary_dilation
import numpy as np
import pdb
import argparse
import sys
import SimpleITK as sitk
from skimage import filters
from skimage.filters import threshold_multiotsu

sys.path.append(os.path.abspath('C:/Users/michael/Projects/Inpainting_Preprocessing/SAB'))

###################################################Segment Any Bone#####################################################

from models.sam import SamPredictor, sam_model_registry
from models.sam.modeling.prompt_encoder import attention_fusion
import numpy as np
import os
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from dsc import dice_coeff
import torchio as tio
import nrrd
import PIL
import cfg
from funcs import *
from predict_funs import *

import torch
print(torch.__version__)
print(torch.cuda.is_available())

###################################################Segment Any Bone#####################################################

def main(nifti_dir_ukbb, offset_mm, finetune_mask, nifti_dir_fastmri=None):
    # Process UKBB
    print(f"Nifti Path UKBB: {nifti_dir_ukbb}")
    nifti_dir_ukbb_raw = os.path.join(nifti_dir_ukbb, 'raw')
    # List all items in the directory and filter only directories
    directories = [d for d in os.listdir(nifti_dir_ukbb_raw) if os.path.isdir(os.path.join(nifti_dir_ukbb_raw, d))]
    len_dir = len(directories)
    print(f'UKBB has {len_dir} different patients')
    len_dir = 0
    split_index = int(len_dir * 0.5)
    process_folders_nifti(nifti_dir_ukbb, nifti_dir_ukbb_raw, split_index, offset_mm=offset_mm, finetune_mask=finetune_mask)

    # Process fastMRI
    if nifti_dir_fastmri:
        print(f"Nifti Path fastMRI: {nifti_dir_fastmri}")
        nifti_dir_fastmri_raw = os.path.join(nifti_dir_fastmri, 'raw')
        directories = [d for d in os.listdir(nifti_dir_fastmri_raw) if os.path.isdir(os.path.join(nifti_dir_fastmri_raw, d))]
        len_dir = len(directories)
        print(f'fastMRI has {len_dir} different patients')
        split_index = int(len_dir * 0.8)
        process_folders_nifti(nifti_dir_fastmri, nifti_dir_fastmri_raw, split_index, offset_mm=offset_mm)
    else:
        print('No fastMRI data provided.')

def process_folders_nifti(output_root, raw_path, split_index, offset_mm, MODE='3D', random_versions=5, finetune_mask=False):
    # Get the list of patient folders in the raw directory
    folder_paths = [os.path.join(raw_path, d) for d in os.listdir(raw_path) if os.path.isdir(os.path.join(raw_path, d))]
    folder_paths.sort()  # Ensure consistent order

    # Split into training and testing folders based on the split_index
    train_folders = folder_paths[:split_index]
    test_folders = folder_paths[split_index:]

    # Define root directories based on the mode (2D or 3D)
    if MODE == '3D':
        train_root = os.path.join(output_root, 'train_3D')
        test_root = os.path.join(output_root, 'test_3D')
        test_root_gt = os.path.join(output_root, 'test_3D_gt')
    elif MODE == '2D':
        train_root = os.path.join(output_root, 'train_2D')
        test_root = os.path.join(output_root, 'test_2D')
        test_root_gt = os.path.join(output_root, 'test_2D_gt')

    # Create directories if they don't exist
    os.makedirs(train_root, exist_ok=True)
    os.makedirs(test_root, exist_ok=True)
    os.makedirs(test_root_gt, exist_ok=True)

    # Process each patient's folder
    for folder in folder_paths:
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith(".nii.gz") and not any(file.endswith(suffix) for suffix in ["_healthy.nii.gz", "_diseased.nii.gz", "_mask.nii.gz"]):
                    nifti_path = os.path.join(root, file)
                    print(f'Preprocessing NIFTI files in {nifti_path}')

                    # Extract patient_folder_name from the directory name
                    parent_folder_name = os.path.basename(os.path.dirname(nifti_path))
                    patient_folder_name = parent_folder_name.split('_')[0] + '_' + parent_folder_name.split('_')[1]

                    # Base filename setup
                    base_filename = f'{patient_folder_name}-{file.split(".")[0]}'

                    # Determine the output path based on the split index
                    if folder in train_folders:
                        output_path_patient = os.path.join(train_root, patient_folder_name)
                    else:
                        output_path_patient = os.path.join(test_root, patient_folder_name)
                        output_path_gt_patient = os.path.join(test_root_gt, patient_folder_name)
                        os.makedirs(output_path_gt_patient, exist_ok=True)

                    os.makedirs(output_path_patient, exist_ok=True)

                    # File names
                    groundtruth_filename = f"{base_filename}_healthy.nii.gz"
                    segmentation_filename = f"{base_filename}_SAB.nii.gz"
                    diseased_filename = f"{base_filename}_diseased.nii.gz"
                    mask_filename = f"{base_filename}_mask.nii.gz"

                    # Process images
                    groundtruth, diseased_volume, mask_volume, segmented_image_numpy, new_affine = preprocess_pipeline(
                        nifti_path, output_root, debug=False, _mode=MODE, random_versions=random_versions, offset_mm=offset_mm, finetune_mask=finetune_mask
                    )

                    print(f'Saving NIFTI files for {patient_folder_name}')

                    # Save groundtruth, diseased volume, and mask volume
                    if folder in train_folders:
                        save_nifti(groundtruth, output_path_patient, groundtruth_filename, new_affine)
                        save_nifti(segmented_image_numpy, output_path_patient, segmentation_filename, new_affine)
                    else:
                        save_nifti(groundtruth, output_path_gt_patient, groundtruth_filename, new_affine)
                        save_nifti(segmented_image_numpy, output_path_gt_patient, segmentation_filename, new_affine)

                    # Save the diseased and mask in the appropriate directory
                    save_nifti(diseased_volume, output_path_patient, diseased_filename, new_affine)
                    save_nifti(mask_volume, output_path_patient, mask_filename, new_affine)


def predictVolume(image_array, lower_percentile, upper_percentile):
    dsc_gt = 0
    # image1_vol = tio.ScalarImage(os.path.join(img_folder,image_name))
    #image1_vol = tio.ScalarImage(img_folder)
    # Ensure the input is a 4D tensor with shape (batch_size, channels, depth, height, width)
    if len(image_array.shape) == 3:
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    image1_vol = tio.ScalarImage(tensor=torch.tensor(image_array, dtype=torch.float))

    print('vol shape: %s vol spacing %s' % (image1_vol.shape, image1_vol.spacing))

    # Define the percentiles
    image_tensor = image1_vol.data
    lower_bound = torch_percentile(image_tensor, lower_percentile)
    upper_bound = torch_percentile(image_tensor, upper_percentile)

    # Clip the data
    image_tensor = torch.clamp(image_tensor, lower_bound, upper_bound)
    # Normalize the data to [0, 1]
    image_tensor = (image_tensor - lower_bound) / (upper_bound - lower_bound)
    image1_vol.set_data(image_tensor)

    mask_vol_numpy = np.zeros(image1_vol.shape)
    id_list = list(range(image1_vol.shape[3]))
    for id in id_list:
        atten_map = pred_attention(image1_vol, vnet, id, device)
        atten_map = torch.unsqueeze(torch.tensor(atten_map), 0).float().to(device)

        ori_img, pred_1, voxel_spacing1, Pil_img1, slice_id1 = evaluate_1_volume_withattention(image1_vol,
                                                                                               sam_fine_tune, device,
                                                                                               slice_id=id,
                                                                                               atten_map=atten_map)
        img1_size = Pil_img1.size
        mask_pred = ((pred_1 > 0) == cls).float().cpu()
        pil_mask1 = Image.fromarray(np.array(mask_pred[0], dtype=np.uint8), 'L').resize(img1_size,
                                                                                        resample=PIL.Image.NEAREST)
        mask_vol_numpy[0, :, :, id] = np.asarray(pil_mask1)

    mask_vol = tio.LabelMap(tensor=torch.tensor(mask_vol_numpy, dtype=torch.int), affine=image1_vol.affine)
    #mask_save_folder = os.path.join(predicted_msk_folder, '/'.join(image_name.split('/')[:-1]))
    #Path(mask_save_folder).mkdir(parents=True, exist_ok=True)
    #mask_vol.save(
    #    os.path.join(mask_save_folder, image_name.split('/')[-1].replace('.nii.gz', '_predicted_SAMatten_paired.nrrd')))
    return mask_vol

def numpy_to_sitk_image(numpy_array, original_image):
    # Create an empty SimpleITK image with the same size as the original
    sitk_image = sitk.GetImageFromArray(numpy_array)

    # Set metadata
    sitk_image.SetSpacing(original_image.GetSpacing())
    sitk_image.SetOrigin(original_image.GetOrigin())
    sitk_image.SetDirection(original_image.GetDirection())

    return sitk_image

def create_3d_ellipsoid(image_shape, radii=(45, 30, 5), position=(128, 140, 20)):
    mask = np.zeros(image_shape, dtype=bool)

    # Create a grid of coordinates
    grid = np.ogrid[[slice(0, i) for i in image_shape]]

    # Calculate the ellipsoid equation
    ellipsoid_equation = sum(((grid[d] - position[d]) / radii[d]) ** 2 for d in range(3))

    # Set the mask to True where the equation is less than or equal to 1
    mask[ellipsoid_equation <= 1] = True

    return mask

def create_patella_offset_mask(binary_mask_patella_array, offset_mask_array, centroid_x):
    # Step 1: Start with the offset_mask_array as the base
    final_mask = np.copy(offset_mask_array)

    # Step 2: Make everything in the binary_mask_patella_array black (False)
    final_mask[binary_mask_patella_array == True] = False

    # Step 3: Make everything above the centroid_x black (False)
    for z in range(final_mask.shape[2]):
        final_mask[:, centroid_x+1:, z] = False

    return final_mask

def process_patella_binary_mask(binary_mask_array, min_voxels=3000, max_voxels=20000, central_region=(16, 25)):
    # Perform connected component analysis

    success = False

    connected_components = sitk.ConnectedComponent(sitk.GetImageFromArray(binary_mask_array))

    # LabelShapeStatistics to compute properties of connected components
    label_shape_stats = sitk.LabelShapeStatisticsImageFilter()
    label_shape_stats.Execute(connected_components)

    # Collect voxel counts and centroids for all connected components
    voxel_counts = {}
    centroids = {}
    for label in label_shape_stats.GetLabels():
        voxel_counts[label] = label_shape_stats.GetNumberOfPixels(label)
        centroids[label] = label_shape_stats.GetCentroid(label)

    # Sort voxel counts in descending order
    sorted_voxel_counts = sorted(voxel_counts.items(), key=lambda x: x[1], reverse=True)

    # Find the patella label based on voxel count and centroid location
    patella_label = None
    for label, count in sorted_voxel_counts:
        if min_voxels <= count <= max_voxels:
            centroid = centroids[label]
            _, centroid_y, centroid_z = np.round(centroid).astype(int)
            if 150 < centroid_y:
                if central_region[0] <= centroid_z <= central_region[1]:
                    patella_label = label
                    print(f'Patella found! centroid: {centroid}')
                    success = True
                    break

    if patella_label is None:
        print("Patella not found!.")
        return (0, 0, 0), np.zeros_like(binary_mask_array), success  # Return default centroid and an empty mask if no patella is found

    # Get the centroid of the identified patella label
    centroid = label_shape_stats.GetCentroid(patella_label)
    centroid_z, centroid_x, centroid_y = np.round(centroid).astype(int)

    # Generate a binary mask for the identified patella label
    patella_mask = sitk.GetArrayFromImage(sitk.Equal(connected_components, patella_label))

    return (centroid_z, centroid_x, centroid_y), patella_mask, success

def compute_anisotropic_offset_surface(binary_mask, offset_mm, voxel_spacing):
    # Convert offset from mm to voxel space for each dimension
    offset_voxels = [int(offset_mm / spacing) for spacing in voxel_spacing]

    # Apply anisotropic dilation (different radius in each direction)
    structuring_element = sitk.BinaryDilate(
        binary_mask,
        [offset_voxels[0], offset_voxels[1], offset_voxels[2]]  # Radii for x, y, and z
    )

    return structuring_element

def preprocess_pipeline(input_path, save_path, offset_mm, lower_percentile=1, upper_percentile=99, debug=False, _mode='3D', random_versions = 0, finetune_mask = False):
    print('[0] START')
    print(f'Processing image: {input_path}')
    img = nib.load(input_path)
    voxel_sizes = img.header.get_zooms()
    original_shape = img.shape
    original_affine = img.affine
    print(f'Original size: {original_shape}')
    print(f'Original Voxel sizes: {voxel_sizes}')

    target_voxel_size = (0.6, 0.6, 4.5)
    zoom_factors = [current / target for target, current in zip(target_voxel_size, voxel_sizes)]

    print('--------SIZE ADJUSTMENTS FOR VOXEL SIZES (256x256x32)--------')
    print("[1] Resample the image ...")
    resampled_data = zoom(img.get_fdata(), zoom_factors, order=3, mode='nearest')
    resampled_data = resampled_data.astype(np.float32)

    new_affine = np.diag([target_voxel_size[0], target_voxel_size[1], target_voxel_size[2], 1])

    if debug:
        resampled_img = nib.Nifti1Image(resampled_data, new_affine)
        save_file_path = os.path.join(save_path, f"resampled_{os.path.basename(input_path)}")
        nib.save(resampled_img, save_file_path)
        print(f"Resampled image saved to: {save_file_path}")

    # Center crop the image if necessary, or pad it to reach the target size

    # Check if any dimension is out of bounds
    if resampled_data.shape[0] > 256 or resampled_data.shape[1] > 256 or resampled_data.shape[2] > 32:
        print("[2] Center crop the image ...")

        # Determine the new shape based on the bounds
        new_shape = (
            min(resampled_data.shape[0], 256),
            min(resampled_data.shape[1], 256),
            min(resampled_data.shape[2], 32)
        )

        # Calculate the start indices for the center crop
        start_x = (resampled_data.shape[0] - new_shape[0]) // 2
        start_y = (resampled_data.shape[1] - new_shape[1]) // 2
        start_z = (resampled_data.shape[2] - new_shape[2]) // 2

        # Center crop the volume
        cropped_data = resampled_data[
                       start_x:start_x + new_shape[0],
                       start_y:start_y + new_shape[1],
                       start_z:start_z + new_shape[2]
                       ]
        print("Volume cropped to shape:", cropped_data.shape)

        if debug:
            # Save the cropped image
            cropped_img = nib.Nifti1Image(cropped_data, new_affine)
            save_file_path = os.path.join(save_path, f"cropped_{os.path.basename(input_path)}")
            nib.save(cropped_img, save_file_path)
            print(f"Cropped image saved to: {save_file_path}")
    else:
        print("No cropping needed. Current shape:", resampled_data.shape)
        cropped_data = resampled_data

    if cropped_data.shape[0] < 256 or cropped_data.shape[1] < 256 or cropped_data.shape[2] < 32:
        print("[3] Padding the image ...")
        # Calculate the padding needed
        desired_shape = (256, 256, 32)

        padding = [
            (max((desired_shape[0] - cropped_data.shape[0]) // 2, 0),
             max((desired_shape[0] - cropped_data.shape[0] + 1) // 2, 0)),
            (max((desired_shape[1] - cropped_data.shape[1]) // 2, 0),
             max((desired_shape[1] - cropped_data.shape[1] + 1) // 2, 0)),
            (max((desired_shape[2] - cropped_data.shape[2]) // 2, 0),
             max((desired_shape[2] - cropped_data.shape[2] + 1) // 2, 0))
        ]
        padding_info = padding
        # Pad the volume with zeros
        padded_data = np.pad(cropped_data, padding, mode='constant', constant_values=0)
        if debug:
            # Save the padded image
            padded_img = nib.Nifti1Image(padded_data, new_affine)
            save_file_path = os.path.join(save_path, f"padded_{os.path.basename(input_path)}")
            nib.save(padded_img, save_file_path)
            print(f"Padded image saved to: {save_file_path}")
    else:
        print("No padding needed. Current shape:", cropped_data.shape)
        padding_info = [(0, 0), (0, 0), (0, 0)]  # Indicates no padding
        padded_data = cropped_data

    print(f'cropped_and_padded_arr size: {padded_data.shape}')

    print('--------INTENSITY ADJUSTMENTS--------')

    print("[4] Clip all values above and below upper and lower percentile...")
    # Clip and normalize the image
    lower_bound = np.percentile(padded_data, lower_percentile)
    upper_bound = np.percentile(padded_data, upper_percentile)
    clipped_data = np.clip(padded_data, lower_bound, upper_bound)

    if debug:
        # Save the clipped image
        clipped_img = nib.Nifti1Image(clipped_data, new_affine)
        save_file_path = os.path.join(save_path, f"clipped_{os.path.basename(input_path)}")
        nib.save(clipped_img, save_file_path)
        print(f"Clipped image saved to: {save_file_path}")

    print("[5] Normalize the image ...")
    out_normalized = (clipped_data - clipped_data.min()) / (clipped_data.max() - clipped_data.min())

    final_shape = out_normalized.shape

    assert final_shape == (256, 256, 32), f'Final shape is not correct: {final_shape}'

    # Apply multi-level Otsu's thresholding to identify different intensity regions
    num_classes = 4  # Number of thresholds you want to apply; adjust as needed
    thresholds = threshold_multiotsu(out_normalized, classes=num_classes)

    # The first threshold defines the separation between the lowest intensity group (background) and the others
    otsu_threshold = thresholds[0]
    otsu_threshold = round(float(otsu_threshold), 2)
    print(f"Otsu threshold after rounding: {otsu_threshold}")

    low_threshold_value = round(float(otsu_threshold), 2)
    # Convert the image to a SimpleITK image for further processing
    sitk_image = sitk.GetImageFromArray(out_normalized)

    print("[5.2] Morphological operations to identify the foreground object (knee)...")

    # Convert the image to a binary image (0 or 1)
    binary_image = sitk.BinaryThreshold(sitk_image, lowerThreshold=low_threshold_value, upperThreshold=1.0,
                                        insideValue=1, outsideValue=0)
    # Perform morphological closing to fill small holes within the object
    closed_image = sitk.BinaryMorphologicalClosing(binary_image, (5, 5, 3))

    # Perform morphological opening to remove small noise in the background
    opened_image = sitk.BinaryMorphologicalOpening(closed_image, (5, 5, 3))

    print("[5.3] Extracting the largest connected component as the knee (foreground)...")
    # Label connected components
    connected_components = sitk.ConnectedComponent(opened_image)
    largest_component = sitk.RelabelComponent(connected_components, sortByObjectSize=True)
    largest_component = sitk.BinaryThreshold(largest_component, 1, 1, 1, 0)  # Keep only the largest component
    # Convert the largest component to a numpy array, representing the foreground (knee)
    foreground = sitk.GetArrayFromImage(largest_component)

    print("[5.4] Apply a median filter to remove small blobs and smooth the foreground...")

    # Apply a Gaussian filter to smooth the edges
    smoothed_foreground = gaussian_filter(foreground.astype(float), sigma=(10,10,2))  # Adjust sigma as needed 3

    # Apply a median filter to clean up small blobs
    filtered_foreground = median_filter(smoothed_foreground, size=(40, 40, 3))  # Adjust size as needed (7, 7, 3)

    # Convert the filtered image back to binary
    foreground = (filtered_foreground > 0.5).astype(np.uint8)  # Adjust threshold if necessary

    if padding_info != [(0, 0), (0, 0), (0, 0)]:  # Check if padding was applied
        # Use padding info to determine regions to zero out
        z_start = padding_info[2][0]
        z_end = padded_data.shape[2] - padding_info[2][1]

        # Apply the mask and keep only relevant slices
        foreground = np.copy(foreground)
        foreground[:, :, :z_start] = 0  # Zero out slices above the padding area
        foreground[:, :, z_end:] = 0  # Zero out slices below the padding area
    '''
    ##PLOTTING##

    # Number of slices
    num_slices = foreground.shape[2]

    # Determine grid size (e.g., 4x8 for 32 slices)
    grid_size = (4, 8)  # Adjust as needed
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(16, 8))

    # Plot each slice
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            slice_index = i * grid_size[1] + j
            if slice_index < num_slices:
                axes[i, j].imshow(foreground[:, :, slice_index], cmap='gray')
                axes[i, j].axis('off')  # Hide axes

    plt.tight_layout()
    plt.show()
    '''
    print("[5.5] Set the background pixels to zero...")

    # Instead of thresholding
    out_normalized[foreground == 0] = 0

    print('--------Segmentation, find the Patella, define the mask--------')

    print("[6] Segment the image with SegmentAnyBone ...")

    segmented_image = predictVolume(out_normalized, lower_percentile=1, upper_percentile=99)

    # Convert the array to NumPy array
    segmented_image_numpy = segmented_image.numpy()

    # Transpose the array to match teh NifTI format (if necessary) --> binary mask as numpy array
    segmented_image_numpy = np.transpose(segmented_image_numpy, (1, 2, 3, 0))  # Adjust the axes order if needed

    # Create a NIfTI image
    affine = np.diag([0.6, 0.6, 4.5, 1])  # for preprocesses images
    nii_img = nib.Nifti1Image(segmented_image_numpy, affine)

    print("[6.1] Connected components & identification of patella centroid")
    # Convert to (256, 256, 32)
    segmented_image_numpy = segmented_image_numpy[:,:,:,0]
    binary_mask_array = np.transpose(segmented_image_numpy, (2, 1, 0))
    binary_mask_sitk = sitk.GetImageFromArray(binary_mask_array)

    print("[7] Find the Patella outline based on segmentation, create a mask around the patella ...")
    #  Perform morphological opening on the binary mask
    binary_mask_opened = sitk.BinaryMorphologicalOpening(binary_mask_sitk, (1, 1, 1))
    # Convert the binary mask back to a NumPy array if needed
    binary_mask_opened_array = sitk.GetArrayFromImage(binary_mask_opened)

    centroid, binary_mask_patella_array, success_ = process_patella_binary_mask(binary_mask_array)

    # Convert to (256, 256, 32)
    voxel_spacing = target_voxel_size[::-1]  # Reverses the order
    #voxel_spacing = np.transpose(target_voxel_size, (2, 1, 0))

    binary_mask_patella_array = np.transpose(binary_mask_patella_array, (2, 1, 0))

    ##SMOOTHING OF PATELLA##

    # Apply Gaussian filter for smoothing
    gaussian_smoothed = gaussian_filter(binary_mask_patella_array.astype(float), sigma=1)  # Adjust sigma as needed closed_mask

    # Apply Median filter to remove small blobs
    median_smoothed = median_filter(gaussian_smoothed, size=(2, 2, 3))  # Adjust size as needed

    # Convert back to binary
    binary_mask_patella_array = (median_smoothed > 0.5).astype(np.uint8)  # Adjust threshold if necessary

    binary_mask_patella = numpy_to_sitk_image(binary_mask_patella_array, binary_mask_sitk)

    offset_surface = compute_anisotropic_offset_surface(binary_mask_patella, offset_mm, voxel_spacing)
    offset_mask_array = sitk.GetArrayFromImage(offset_surface)
    offset_mask_array = np.transpose(offset_mask_array, (2, 1, 0))

    mask = create_patella_offset_mask(np.transpose(binary_mask_patella_array, (2, 1, 0)),
                                      offset_mask_array, centroid[1] + 5)

    mask = np.transpose(mask, (2, 1, 0))

    mask = mask.astype(bool)

    if success_==False:
        mask = create_3d_ellipsoid(out_normalized.shape)


    print("[8] Create the mask and diseased_image around the patella (or not in TD region for finetuning)...")

    #print('shape of mask', mask.shape)
    diseased_volume = np.copy(out_normalized)
    #print('shape of diseased_volume', diseased_volume.shape)
    diseased_volume[mask] = 0  # Apply mask to volume

    mask_volume = np.zeros_like(out_normalized)
    mask_volume[mask] = 1
    groundtruth = out_normalized

    print("[9] Further process segmentation...")
    # Convert the segmented image to a SimpleITK image
    sitk_segmented_image = sitk.GetImageFromArray(segmented_image_numpy)

    # Extract connected components
    connected_components = sitk.ConnectedComponent(sitk_segmented_image)

    # Relabel the connected components based on size
    relabeled_components = sitk.RelabelComponent(connected_components, sortByObjectSize=True)

    # Get the number of connected components
    num_components = sitk.GetArrayFromImage(relabeled_components).max()

    # Keep only the three largest components
    keep_labels = list(range(1, min(4, num_components + 1)))  # Labels start from 1

    # Create a mask for the three largest components
    three_largest_mask = sitk.BinaryThreshold(relabeled_components, lowerThreshold=min(keep_labels),
                                              upperThreshold=max(keep_labels), insideValue=1, outsideValue=0)

    # Convert mask to numpy array
    three_largest_mask_array = sitk.GetArrayFromImage(three_largest_mask)

    if padding_info != [(0, 0), (0, 0), (0, 0)]:  # Check if padding was applied
        # Use padding info to determine regions to zero out
        z_start = padding_info[2][0]
        z_end = padded_data.shape[2] - padding_info[2][1]

        # Apply the mask and keep only relevant slices
        cleaned_segmented_image_numpy = np.copy(segmented_image_numpy)
        cleaned_segmented_image_numpy[:, :, :z_start] = 0  # Zero out slices above the padding area
        cleaned_segmented_image_numpy[:, :, z_end:] = 0  # Zero out slices below the padding area

    else:
        # No padding was applied, so keep the entire volume
        cleaned_segmented_image_numpy = np.copy(segmented_image_numpy)

    # Apply the mask to remove small blobs
    cleaned_segmented_image_numpy *= three_largest_mask_array

    segmented_image_numpy = cleaned_segmented_image_numpy

    return groundtruth, diseased_volume, mask_volume, segmented_image_numpy, new_affine

def save_nifti(data, path, filename, affine):
    nib.save(nib.Nifti1Image(data, affine), os.path.join(path, filename))


if __name__ == "__main__":

    print('Settings for SegmentAnyBone (SAB)...')
    args = cfg.parse_args()
    from monai.networks.nets import VNet

    args.if_mask_decoder_adapter = True
    args.if_encoder_adapter = True
    args.decoder_adapt_depth = 2

    rgs = cfg.parse_args()
    args.if_mask_decoder_adapter = True
    args.if_encoder_adapter = True
    args.decoder_adapt_depth = 2

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(device)

    checkpoint_directory = './SAB/checkpoints'  # path to your checkpoint
    #img_folder = os.path.join('images')
    #gt_msk_folder = os.path.join('masks')
    #predicted_msk_folder = os.path.join('predicted_masks')
    cls = 1

    sam_fine_tune = sam_model_registry["vit_t"](args, checkpoint=os.path.join('mobile_sam.pt'), num_classes=2)
    sam_fine_tune.attention_fusion = attention_fusion()
    sam_fine_tune.load_state_dict(
        torch.load(os.path.join(checkpoint_directory, 'bone_sam.pth'), map_location=torch.device(device)), strict=True)
    sam_fine_tune = sam_fine_tune.to(device).eval()

    vnet = VNet().to(device)
    model_directory = "./SAB/model_dir"
    vnet.load_state_dict(torch.load(os.path.join(model_directory, 'atten.pth'), map_location=torch.device(device)))


    print('STEP 2: Preprocess NIFTI files for Diffusion Model')
    parser = argparse.ArgumentParser(description="Preprocess NIFTI files for Diffusion Model")

    parser.add_argument('--nifti_dir_ukbb', type=str, required=True,
                        help="UKBB nifti directory name.")
    parser.add_argument('--nifti_dir_fastmri', type=str, required=False,
                        help="fastMRI nifti directory name.")
    parser.add_argument('--offset_mm', type=int, required=True,
                        help="Offset in mm for the mask around the patella (thickness).")
    parser.add_argument('--finetune_mask', type=bool, required=True,
                        help="A 3d ellipsoid mask will be generated (not within the trochlear region) instead of the patella mask. This can be used to fine-tune the WDM to the UKBB dataset.")

    args = parser.parse_args()

    main(nifti_dir_ukbb=args.nifti_dir_ukbb,
         nifti_dir_fastmri=args.nifti_dir_fastmri, offset_mm=args.offset_mm, finetune_mask=args.finetune_mask)