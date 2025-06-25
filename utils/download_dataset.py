import os
import urllib.request
import zipfile
import shutil

def download_indian_pines():
    """Download and extract Indian Pines dataset"""
    # Create datasets directory
    os.makedirs('datasets', exist_ok=True)
    
    # Download URL
    url = 'http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat'
    gt_url = 'http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat'
    
    # Download files
    print("Downloading Indian Pines dataset...")
    urllib.request.urlretrieve(url, 'datasets/Indian_pines_corrected.mat')
    urllib.request.urlretrieve(gt_url, 'datasets/Indian_pines_gt.mat')
    print("Download complete!")

if __name__ == '__main__':
    download_indian_pines() 