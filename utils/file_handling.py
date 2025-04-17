"""
File handling utilities for OMRChecker API
"""
import os
import shutil
import time
import uuid
import datetime
from pathlib import Path
import uuid

from src.logger import logger

def transform_result_format(result_data):
    """Transform flat result keys (q1, q2, etc.) into structured format with answers array"""
    if not result_data or not isinstance(result_data, dict):
        return result_data
    
    metadata = {k: v for k, v in result_data.items() 
               if not k.startswith('q') or not k[1:].replace('-', '').isdigit()}
    
    answers = []
    for key, value in result_data.items():
        if key.startswith('q') and key[1:].replace('-', '').isdigit():
            answers.append({
                "key": key,
                "value": value
            })
    
    transformed_result = metadata.copy()
    transformed_result['answers'] = answers
    
    return transformed_result

def save_to_public_images(image_path, prefix, api_host, public_images_dir):
    """Save image to public directory with unique filename and return its URL"""
    try:
        timestamp = int(time.time())
        unique_id = str(uuid.uuid4())[:8]
        file_ext = os.path.splitext(image_path)[1].lower()
        
        filename = f"{prefix}_{timestamp}_{unique_id}{file_ext}"
        public_path = os.path.join(public_images_dir, filename)
        
        shutil.copy2(image_path, public_path)
        
        return f"{api_host}/images/{filename}"
    except Exception as e:
        logger.error(f"Failed to save public image: {e}")
        return None

def clean_folder_contents(folder_path, folder_name):
    """Delete all contents inside the folder without deleting the folder itself"""
    try:
        logger.info(f"Cleaning {folder_name} directory at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        items_count = 0
        
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isfile(item_path):
                try:
                    os.remove(item_path)
                    items_count += 1
                except Exception as e:
                    logger.error(f"Error deleting file {item_path}: {str(e)}")
            elif os.path.isdir(item_path):
                try:
                    shutil.rmtree(item_path)
                    items_count += 1
                except Exception as e:
                    logger.error(f"Error removing directory {item_path}: {str(e)}")
                    
        logger.info(f"Cleaned {items_count} items from {folder_name} directory")
        return items_count
    except Exception as e:
        logger.error(f"Error during scheduled cleaning of {folder_name} directory: {str(e)}")
        return 0

def clean_all_folders(config):
    """Delete contents of all directories: inputs, outputs, images"""
    total_items = 0
    
    total_items += clean_folder_contents(config['INPUTS_DIR_ABS'], "inputs")
    total_items += clean_folder_contents(config['OUTPUTS_DIR_ABS'], "outputs")
    total_items += clean_folder_contents(config['PUBLIC_IMAGES_DIR_ABS'], "images")
    
    logger.info(f"Cleaned a total of {total_items} items from all directories")
    return total_items 