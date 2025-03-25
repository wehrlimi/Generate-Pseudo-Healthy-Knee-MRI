import os
import pydicom
import pandas as pd
import xml.etree.ElementTree as ET


def extract_descriptions(xml_path):
    """Extracts series descriptions from a CONTENT.XML file, considering the study ID."""
    description_map = {}
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for request in root.findall('.//request'):
        for study in request.findall('.//study'):
            study_id = study.get('id')
            for series in study.findall('.//series'):
                series_id = series.get('id')
                description = series.find('.//description').text if series.find(
                    './/description') is not None else 'NoDescription'
                unique_id = f"{study_id}_{series_id}"
                description_map[unique_id] = description
    return description_map


def process_all_xml(patient_folders_root):
    all_descriptions = {}
    for root_dir, dirs, files in os.walk(patient_folders_root):
        for file in files:
            if file.upper() == "CONTENT.XML":
                xml_path = os.path.join(root_dir, file)
                all_descriptions.update(extract_descriptions(xml_path))
    return all_descriptions


def extract_dicom_metadata(dicom_file, patient_name, subdir, description):
    """Extract metadata from a DICOM file, including additional fields."""
    try:
        ds = pydicom.dcmread(dicom_file, stop_before_pixels=True)
        metadata = {
            'PatientFolder': patient_name,
            'FullFolderPath': subdir,
            'Filename': os.path.basename(dicom_file),
            'PatientID': ds.get('PatientID', 'N/A'),
            'StudyDate': ds.get('StudyDate', 'N/A'),
            'Modality': ds.get('Modality', 'N/A'),
            'SeriesDescription': ds.get('SeriesDescription', 'N/A'),
            'Manufacturer': ds.get('Manufacturer', 'N/A'),
            'SeriesNumber': ds.get('SeriesNumber', 'N/A'),
            'InstanceNumber': ds.get('InstanceNumber', 'N/A'),
            'TeslaFieldStrength': ds.get('MagneticFieldStrength', 'N/A'),
            'RepetitionTime': ds.get('RepetitionTime', 'N/A'),
            'EchoTime': ds.get('EchoTime', 'N/A'),
            'SequenceName': ds.get('SequenceName', 'N/A'),
            'BodyPartExamined': ds.get('BodyPartExamined', 'N/A'),
            'SequenceVariant': ds.get('SequenceVariant', 'N/A'),
            'MRAcquisitionType': ds.get('MRAcquisitionType', 'N/A'),
            'SpacingBetweenSlices': ds.get('SpacingBetweenSlices', 'N/A'),
            'SoftwareVersions': ds.get('SoftwareVersions', 'N/A'),
            'AcquisitionMatrix': ds.get('AcquisitionMatrix', 'N/A'),
            'PatientPosition': ds.get('PatientPosition', 'N/A'),
            'ScanOptions': ds.get('ScanOptions', 'N/A'),
            'Sequence': determine_sequence_type(ds),
            'Laterality': ds.get('Laterality', 'N/A'),
            'XMLSeriesDescription': description  # Include the series description from CONTENT.XML
        }
        return metadata
    except Exception as e:
        print(f"Error reading {dicom_file}: {e}")
        return None


def determine_sequence_type(ds):
    """Determine the sequence type (PD, T2, T1) based on TR and TE."""
    try:
        tr = float(ds.get('RepetitionTime', 0))
        te = float(ds.get('EchoTime', 0))
        if tr > 2000 and te < 30:
            return 'PD'
        elif te < 30:
            return 'other'
    except:
        return 'Unknown'


def collect_metadata_from_folder(root_folder, mode, descriptions=None):
    """Traverse through folders and collect metadata from the first DICOM file in each series."""
    metadata_list = []
    file_count = 0

    if mode == "fastMRI":
        for subdir, _, files in os.walk(root_folder):
            dicom_files = sorted([file for file in files if file.lower().endswith('.dcm')])
            if dicom_files:
                file_count += 1
                dicom_file_path = os.path.join(subdir, dicom_files[0])
                metadata = extract_dicom_metadata(dicom_file_path, 'N/A', subdir, 'N/A')
                if metadata:
                    metadata_list.append(metadata)
                    print(f"Processed {file_count} files: {dicom_file_path}")
    elif mode == "UKBB":
        for patient_folder in os.listdir(root_folder):
            patient_path = os.path.join(root_folder, patient_folder)
            if os.path.isdir(patient_path):
                dicom_folder = os.path.join(patient_path, 'DICOM')
                if os.path.exists(dicom_folder):
                    for subdir, _, files in os.walk(dicom_folder):
                        dicom_files = sorted([file for file in files])
                        if dicom_files:
                            file_count += 1
                            dicom_file_path = os.path.join(subdir, dicom_files[0])
                            # Generate the unique ID for the series description
                            parts = subdir.split(os.sep)
                            unique_id = f"{parts[-2]}_{parts[-1]}"
                            description = descriptions.get(unique_id,
                                                           'NoDescription') if descriptions else 'NoDescription'
                            metadata = extract_dicom_metadata(dicom_file_path, patient_folder, subdir, description)
                            if metadata:
                                metadata_list.append(metadata)
                                print(f"Processed {file_count} files: {dicom_files}")
    else:
        print("Invalid mode. Please select either 'fastMRI' or 'UKBB'.")
        return []

    return metadata_list



def save_metadata_to_excel(metadata_dict, output_file):
    """
    Save the collected metadata to an Excel file with multiple tabs.

    Parameters:
    - metadata_dict: Dictionary where keys are sheet names and values are lists of metadata.
    - output_file: Path to the output Excel file.
    """
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        for sheet_name, metadata_list in metadata_dict.items():
            df = pd.DataFrame(metadata_list)
            df.to_excel(writer, sheet_name=sheet_name, index=False)


def main():
    root_folder_fastMRI = r'C:\Users\michael\Documents\Data\fastmri_knee_batch2_rawdata'
    root_folder_UKBB = r'C:\Users\michael\Documents\Data\UKBB_rawdata'
    output_file = r'C:\Users\michael\Documents\Data\dicom_metadata.xlsx'

    metadata_list_fastMRI = collect_metadata_from_folder(root_folder_fastMRI, "fastMRI")

    # Process descriptions for UKBB mode
    descriptions = process_all_xml(root_folder_UKBB)
    metadata_list_UKBB = collect_metadata_from_folder(root_folder_UKBB, "UKBB", descriptions)

    # Create a dictionary of metadata lists and corresponding sheet names
    metadata_dict = {
        'fastMRI': metadata_list_fastMRI,
        'UKBB': metadata_list_UKBB,
    }

    save_metadata_to_excel(metadata_dict, output_file)
    print(f"Metadata has been saved to {output_file}")


if __name__ == "__main__":
    main()

