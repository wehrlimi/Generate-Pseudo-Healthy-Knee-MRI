# Output message
Write-Output "RUN BASH"

# Define variables
$EXCEL_PATH = "C:/Users/michael/Documents/Data/Fresh/DICOM_metadata_UKBB_fastMRI_Analysis.xlsx"
$SHEET_NAME_UKBB = "UKBB_selected"
$SHEET_NAME_FASTMRI = "fastMRI_selected"
$NIFTI_DIR_UKBB = "C:/Users/michael/Documents/Data/Fresh/UKBB_nifti"
$NIFTI_DIR_FASTMRI = "C:/Users/michael/Documents/Data/Fresh/fastMRI_nifti"

$OFFSET_MM = 30 #[mm]
$FINETUNE_MASK = $True

# Step 1: Convert interested DICOM folders to NIFTI in the correct naming
# Activate the first conda environment
Write-Output "ACTIVATE CONDA ENVIRONMENT FOR STEP 1"
conda activate env_step1_SAB
Write-Output "RUN STEP 1"
python step1_convert_D2N.py `
    "--excel_path=$EXCEL_PATH" `
    "--sheet_name_ukbb=$SHEET_NAME_UKBB" `
    "--sheet_name_fastmri=$SHEET_NAME_FASTMRI" `
    "--nifti_dir_ukbb=$NIFTI_DIR_UKBB" `
    "--nifti_dir_fastmri=$NIFTI_DIR_FASTMRI"

# Optionally deactivate the first conda environment
conda deactivate

Write-Output "ACTIVATE CONDA ENVIRONMENT FOR STEP 2"
# Step 2: Preprocess for WDM 3D with SAB
conda activate env_step2_SAB # Environment with SAB & necessary things for preprocessing
python step2_preprocessing_SAB.py `
    "--nifti_dir_ukbb=$NIFTI_DIR_UKBB" `
    "--nifti_dir_fastmri=$NIFTI_DIR_FASTMRI" `
    "--offset_mm=$OFFSET_MM" `
    "--finetune_mask=$($FINETUNE_MASK -eq $True)"

