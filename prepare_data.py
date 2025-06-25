import os
import numpy as np
from scipy.io import loadmat
import urllib.request
import zipfile
import shutil
import time
import requests
from tqdm import tqdm

def download_file(url, filename, max_retries=3):
    """Download a file from URL with retries and progress bar"""
    for attempt in range(max_retries):
        try:
            print(f'Downloading {filename} (attempt {attempt + 1}/{max_retries})...')
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte
            
            with open(filename, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(block_size):
                    size = f.write(data)
                    pbar.update(size)
            return True
        except Exception as e:
            print(f'Error downloading {filename}: {str(e)}')
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retrying
            continue
    return False

def extract_zip(zip_path, extract_to):
    """Extract a zip file"""
    print(f'Extracting {zip_path}...')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def prepare_indian_pines():
    """Prepare Indian Pines dataset"""
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Define possible source file names
    possible_data_files = [
        r"D:\Projects\HSI\data\Indian_pines.mat",
        r"D:\Projects\HSI\data\Indian_pines_corrected.mat",
        r"D:\Projects\HSI\data\indian_pines.mat",
        r"D:\Projects\HSI\data\indian_pines_corrected.mat"
    ]
    possible_gt_files = [
        r"D:\Projects\HSI\data\Indian_pines_gt.mat",
        r"D:\Projects\HSI\data\indian_pines_gt.mat"
    ]
    
    # Find existing data file
    source_data = None
    for data_file in possible_data_files:
        if os.path.exists(data_file):
            source_data = data_file
            break
    
    # Find existing ground truth file
    source_gt = None
    for gt_file in possible_gt_files:
        if os.path.exists(gt_file):
            source_gt = gt_file
            break
            
    # Check if files were found
    if source_data is None:
        raise FileNotFoundError(
            "Indian Pines data file not found. Tried: \n" + 
            "\n".join(f"- {f}" for f in possible_data_files)
        )
    if source_gt is None:
        raise FileNotFoundError(
            "Indian Pines ground truth file not found. Tried: \n" + 
            "\n".join(f"- {f}" for f in possible_gt_files)
        )
      # Load and process data directly
    print("Loading and processing Indian Pines files...")
    print(f"Reading data file: {source_data}")
    print(f"Reading ground truth file: {source_gt}")
    
    try:
        # Load data
        mat_data = loadmat(source_data)
        data = mat_data['indian_pines_corrected']
        
        # Load ground truth
        mat_gt = loadmat(source_gt)
        labels = mat_gt['indian_pines_gt']
        
        # Save processed data
        print("Saving processed data...")
        np.save('data/indian_pines.npy', data)
        np.save('data/indian_pines_labels.npy', labels)
        
    except Exception as e:
        print(f"Error processing files: {str(e)}")
        raise

def prepare_pavia_university():
    """Prepare Pavia University dataset"""
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Define possible source file names
    possible_data_files = [
        r"D:\Projects\HSI\data\PaviaU.mat",
        r"D:\Projects\HSI\data\Pavia.mat",
        r"D:\Projects\HSI\data\pavia_university.mat",
        r"D:\Projects\HSI\data\paviaU.mat"
    ]
    possible_gt_files = [
        r"D:\Projects\HSI\data\PaviaU_gt.mat",
        r"D:\Projects\HSI\data\Pavia_gt.mat",
        r"D:\Projects\HSI\data\pavia_university_gt.mat",
        r"D:\Projects\HSI\data\paviaU_gt.mat"
    ]
    
    # Find existing data file
    source_data = None
    for data_file in possible_data_files:
        if os.path.exists(data_file):
            source_data = data_file
            break
    
    # Find existing ground truth file
    source_gt = None
    for gt_file in possible_gt_files:
        if os.path.exists(gt_file):
            source_gt = gt_file
            break
            
    # Check if files were found
    if source_data is None:
        raise FileNotFoundError(
            "Pavia University data file not found. Tried: \n" + 
            "\n".join(f"- {f}" for f in possible_data_files)
        )
    if source_gt is None:
        raise FileNotFoundError(
            "Pavia University ground truth file not found. Tried: \n" + 
            "\n".join(f"- {f}" for f in possible_gt_files)
        )
      # Load and process data directly
    print("Loading and processing Pavia University files...")
    print(f"Reading data file: {source_data}")
    print(f"Reading ground truth file: {source_gt}")
    
    try:
        # Load data
        mat_data = loadmat(source_data)
        data = mat_data['paviaU']
        
        # Load ground truth
        mat_gt = loadmat(source_gt)
        labels = mat_gt['paviaU_gt']
        
        # Save processed data
        print("Saving processed data...")
        np.save('data/pavia_university.npy', data)
        np.save('data/pavia_university_labels.npy', labels)
        
    except Exception as e:
        print(f"Error processing files: {str(e)}")
        raise

def main():
    print('Preparing datasets...')
    
    try:
        # Prepare Indian Pines
        print('\nPreparing Indian Pines dataset...')
        prepare_indian_pines()
        
        # Prepare Pavia University
        print('\nPreparing Pavia University dataset...')
        prepare_pavia_university()
        
        print('\nDataset preparation completed!')
        print('Data files are saved in the data/ directory:')
        print('- indian_pines.npy')
        print('- indian_pines_labels.npy')
        print('- pavia_university.npy')
        print('- pavia_university_labels.npy')
        
    except Exception as e:
        print(f'\nError: {str(e)}')
        print('\nPlease ensure the following files exist:')
        print('1. Indian Pines:')
        print('   - D:\\Projects\\HSI\\data\\Indian_pines.mat')
        print('   - D:\\Projects\\HSI\\data\\Indian_pines_gt.mat')
        print('2. Pavia University:')
        print('   - D:\\Projects\\HSI\\data\\PaviaU.mat')
        print('   - D:\\Projects\\HSI\\data\\PaviaU_gt.mat')

if __name__ == '__main__':
    main() 