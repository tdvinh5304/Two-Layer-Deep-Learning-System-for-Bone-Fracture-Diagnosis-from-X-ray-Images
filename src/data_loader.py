import pandas as pd
import requests
from PIL import Image
import io
from typing import Optional, List, Tuple
import numpy as np


class DataLoader:
    """Class to load and manage X-ray image data"""
    
    def __init__(self, train_image_path: str, train_labeled_studies: str,
                 valid_image_path: str, valid_labeled_studies: str):
        """
        Initialize DataLoader with paths to CSV files
        
        Args:
            train_image_path: Path to CSV containing training image URLs
            train_labeled_studies: Path to CSV containing training labels
            valid_image_path: Path to CSV containing validation image URLs
            valid_labeled_studies: Path to CSV containing validation labels
        """
        try:
            self.train_images = pd.read_csv(train_image_path)
            self.train_labels = pd.read_csv(train_labeled_studies)
            self.valid_images = pd.read_csv(valid_image_path)
            self.valid_labels = pd.read_csv(valid_labeled_studies)
            print("✓ All CSV files loaded successfully")
        except Exception as e:
            print(f"Error loading CSV files: {e}")
            raise
    
    def load_image_from_url(self, url: str) -> Optional[Image.Image]:
        """
        Load image from URL
        
        Args:
            url: URL or path to the image
            
        Returns:
            PIL Image object or None if failed
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert('RGB')
            return image
        except Exception as e:
            print(f"Error loading image from {url}: {e}")
            return None
    
    def get_train_data(self, limit: Optional[int] = None) -> List[Tuple[Image.Image, dict]]:
        """
        Get training data
        
        Args:
            limit: Maximum number of samples to load
            
        Returns:
            List of (image, label_dict) tuples
        """
        data = []
        max_samples = limit if limit else len(self.train_images)
        
        for idx in range(min(max_samples, len(self.train_images))):
            image_url = self.train_images.iloc[idx]['image_url']  # Adjust column name as needed
            image = self.load_image_from_url(image_url)
            
            if image is not None:
                label_info = self.train_labels.iloc[idx].to_dict()
                data.append((image, label_info))
        
        return data
    
    def get_valid_data(self, limit: Optional[int] = None) -> List[Tuple[Image.Image, dict]]:
        """
        Get validation data
        
        Args:
            limit: Maximum number of samples to load
            
        Returns:
            List of (image, label_dict) tuples
        """
        data = []
        max_samples = limit if limit else len(self.valid_images)
        
        for idx in range(min(max_samples, len(self.valid_images))):
            image_url = self.valid_images.iloc[idx]['image_url']  # Adjust column name as needed
            image = self.load_image_from_url(image_url)
            
            if image is not None:
                label_info = self.valid_labels.iloc[idx].to_dict()
                data.append((image, label_info))
        
        return data
    
    def get_dataset_info(self) -> dict:
        """
        Get information about the dataset
        
        Returns:
            Dictionary with dataset statistics
        """
        return {
            'train_images_count': len(self.train_images),
            'train_labels_count': len(self.train_labels),
            'valid_images_count': len(self.valid_images),
            'valid_labels_count': len(self.valid_labels),
            'train_columns': list(self.train_images.columns),
            'label_columns': list(self.train_labels.columns)
        }


if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader(
        "data/train/train_image_paths.csv",
        "data/train/train_labeled_studies.csv",
        "data/valid/valid_image_paths.csv",
        "data/valid/valid_labeled_studies.csv"
    )
    
    print("Dataset Info:")
    print(loader.get_dataset_info())
