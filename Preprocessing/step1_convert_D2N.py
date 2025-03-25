import os
import pandas as pd
import dicom2nifti
import argparse


def main(excel_path, sheet_name_ukbb=None, nifti_dir_ukbb=None, sheet_name_fastmri=None, nifti_dir_fastmri=None):
    # Your script logic goes here

    # Check if UKBB data is provided
    if sheet_name_ukbb and nifti_dir_ukbb:
        print(f"Processing UKBB data...")
        print(f"Excel Path: {excel_path}")
        print(f"Sheet Name UKBB: {sheet_name_ukbb}")
        print(f"Nifti Path UKBB: {nifti_dir_ukbb}")

        # Perform conversion from DICOM to NIFTI for UKBB
        selected_folders_ukbb = read_excel_file(excel_path, sheet_name_ukbb)
        process_selected_dicom_folders(selected_folders_ukbb, nifti_dir_ukbb, sheet_name_ukbb)
    else:
        print("UKBB data not provided.")

    # Check if fastMRI data is provided
    if sheet_name_fastmri and nifti_dir_fastmri:
        print(f"Processing fastMRI data...")
        print(f"Sheet Name fastMRI: {sheet_name_fastmri}")
        print(f"Nifti Path fastMRI: {nifti_dir_fastmri}")

        # Perform conversion from DICOM to NIFTI for fastMRI
        selected_folders_fastmri = read_excel_file(excel_path, sheet_name_fastmri)
        process_selected_dicom_folders(selected_folders_fastmri, nifti_dir_fastmri, sheet_name_fastmri)
    else:
        print("fastMRI data not provided.")


def read_excel_file(excel_path, sheet_name):
    # Read the Excel file
    df = pd.read_excel(excel_path, sheet_name=sheet_name, engine='openpyxl')

    # Drop rows where 'FullFolderPath' is NaN or not a string
    df = df.dropna(subset=['FullFolderPath'])
    df = df[df['FullFolderPath'].apply(lambda x: isinstance(x, str))]

    if 'UKBB' and 'selected' in sheet_name:
        # Filter for UKBB data and set the description
        df = df.dropna(subset=['XMLSeriesDescription'])
        df = df[df['XMLSeriesDescription'].apply(lambda x: isinstance(x, str))]
        df['Description'] = df['XMLSeriesDescription']
    elif 'fastMRI' and 'selected' in sheet_name:
        # Filter for fastMRI data and set the description
        df = df.dropna(subset=['SeriesDescription'])
        df = df[df['SeriesDescription'].apply(lambda x: isinstance(x, str))]
        df['Description'] = df['SeriesDescription']

    return df[['FullFolderPath', 'Filename', 'Description']].to_dict('records')


def convert_dicom_to_nifti(dicom_folder_path, output_path):
    """
    Convert DICOM files to NIfTI files using dicom2nifti library.
    :param dicom_folder_path: Directory containing the DICOM files
    :param output_file_path: Path to save the NIfTI files
    :return: None
    """
    # Ensure output path exists
    os.makedirs(output_path, exist_ok=True)

    # Check if the DICOM directory exists
    if os.path.exists(dicom_folder_path):
        print("DICOM directory exists.")
    else:
        print("DICOM directory does not exist.")

    try:
        # Convert the DICOM folder to NIfTI format
        dicom2nifti.convert_directory(dicom_folder_path, output_path)

    except Exception as e:
        # Handle conversion failure
        print(f'Failed to convert DICOM files in {dicom_folder_path}. Error: {e}')


def process_selected_dicom_folders(selected_folders, output_root, sheet_name, trainsplitt=0.8, MODE='3D'):
    study_folder_map = {}
    study_counter = 1

    # Step 1: Collect the folder paths
    folder_paths = []
    for folder_info in selected_folders:
        dicom_folder = folder_info['FullFolderPath']
        parts = dicom_folder.split(os.sep)

        if 'UKBB' in sheet_name:
            # Create patient folder name for UKBB data
            patient_folder_name = parts[-6] + '_' + parts[-1]
        elif 'fastMRI' in sheet_name:
            # Create patient folder name for fastMRI data
            parent_folder = parts[-3]
            if parent_folder not in study_folder_map:
                study_folder_map[parent_folder] = f'{study_counter:04d}'
                study_counter += 1
            patient_folder_name = study_folder_map[parent_folder]

        # Construct the new output path
        output_path_raw = os.path.join(output_root, 'raw')
        output_path = os.path.join(output_path_raw, f'{patient_folder_name}_rawnifti')
        folder_paths.append((dicom_folder, output_path, patient_folder_name))

    '''
    # Step 2: Split the data into training and testing sets (currently commented out)
    split_index = int(len(folder_paths) * trainsplitt)  # folder_paths
    train_folders = folder_paths[:split_index]
    test_folders = folder_paths[split_index:]

    # Create train and test directories if they don't exist
    if MODE == '3D':
        train_root = os.path.join(output_root, 'train_3D')
        test_root = os.path.join(output_root, 'test_3D')
        test_root_gt = os.path.join(output_root, 'test_3D_gt')
    elif MODE == '2D':
        test_root = os.path.join(output_root, 'test_2D')
        train_root = os.path.join(output_root, 'train_2D')
        test_root_gt = os.path.join(output_root, 'test_2D_gt')

    os.makedirs(train_root, exist_ok=True)
    os.makedirs(test_root, exist_ok=True)
    os.makedirs(test_root_gt, exist_ok=True)
    '''
    # Step 3: Process the folders (train and test sets are skipped for now)
    print('Converting all valid dicom files...')
    process_folders(folder_paths)

    # print('Processing test set...')
    # process_folders(test_folders)


def process_folders(folders):
    i = 0
    # Iterate over all selected folders
    for dicom_folder, output_path, patient_folder_name in folders:
        try:
            # Convert each DICOM folder to NIfTI
            convert_dicom_to_nifti(dicom_folder, output_path)
            print(f'Converted DICOM files in {dicom_folder} to NIFTI in {output_path}...')
            i += 1

        except Exception as e:
            # Handle conversion failures
            print(f'Failed to convert DICOM files in {dicom_folder}. Error: {e}')

    print(f'From {len(folders)} dicom files, {i} were successfully converted to NIFTI format.')


if __name__ == "__main__":
    print('STEP 1: Convert DICOM files to NIfTI format')

    parser = argparse.ArgumentParser(description="Convert DICOM folders to NIFTI format.")

    # Define command-line arguments
    parser.add_argument('--excel_path', type=str, required=True,
                        help="Path to the Excel file containing metadata.")
    parser.add_argument('--sheet_name_ukbb', type=str, required=False,
                        help="Sheet name in Excel for UKBB selected data.")
    parser.add_argument('--sheet_name_fastmri', type=str, required=False,
                        help="Sheet name in Excel for fastMRI selected data.")
    parser.add_argument('--nifti_dir_ukbb', type=str, required=False,
                        help="UKBB NIFTI directory name.")
    parser.add_argument('--nifti_dir_fastmri', type=str, required=False,
                        help="fastMRI NIFTI directory name.")

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with the provided arguments
    main(excel_path=args.excel_path,
         sheet_name_ukbb=args.sheet_name_ukbb,
         nifti_dir_ukbb=args.nifti_dir_ukbb,
         sheet_name_fastmri=args.sheet_name_fastmri,
         nifti_dir_fastmri=args.nifti_dir_fastmri)
