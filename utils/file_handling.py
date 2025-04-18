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
import requests
from urllib.parse import urlparse
import urllib3
from urllib.parse import quote
from urllib.parse import unquote

from src.logger import logger

# Disable insecure request warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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
        # Ensure public images directory exists
        os.makedirs(public_images_dir, exist_ok=True)
        
        timestamp = int(time.time())
        unique_id = str(uuid.uuid4())[:8]
        file_ext = os.path.splitext(image_path)[1].lower()
        
        filename = f"{prefix}_{timestamp}_{unique_id}{file_ext}"
        public_path = os.path.join(public_images_dir, filename)
        
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(public_path), exist_ok=True)
        
        # Log paths for debugging
        logger.info(f"Copying from {image_path} to {public_path}")
        
        # Check if source exists
        if not os.path.exists(image_path):
            logger.error(f"Source file does not exist: {image_path}")
            return None
            
        shutil.copy2(image_path, public_path)
        
        result_url = f"{api_host}/images/{filename}"
        logger.info(f"Created public image URL: {result_url}")
        return result_url
    except Exception as e:
        logger.error(f"Failed to save public image: {e}")
        logger.error(f"Source: {image_path}, Destination dir: {public_images_dir}")
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

def clean_all_folders(app_config):
    """Clean all data directories"""
    total_items = 0
    
    # Clean inputs directory
    inputs_dir = Path(app_config['INPUTS_DIR_ABS'])
    if inputs_dir.exists():
        items = list(inputs_dir.glob("**/*"))
        total_items += len(items)
        shutil.rmtree(inputs_dir)
        os.makedirs(inputs_dir)
        
    # Clean outputs directory
    outputs_dir = Path(app_config['OUTPUTS_DIR_ABS'])
    if outputs_dir.exists():
        items = list(outputs_dir.glob("**/*"))
        total_items += len(items)
        shutil.rmtree(outputs_dir)
        os.makedirs(outputs_dir)
    
    # Clean public images directory
    public_images_dir = Path(app_config['PUBLIC_IMAGES_DIR_ABS'])
    if public_images_dir.exists():
        items = list(public_images_dir.glob("*"))
        total_items += len(items)
        for item in items:
            try:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            except Exception as e:
                logger.warning(f"Error removing {item}: {str(e)}")
    
    return total_items

def download_file_from_url(url, target_dir, app_config=None):
    """
    Download a file from a URL to the target directory
    If URL starts with API_HOST, read directly from filesystem
    
    Args:
        url: The URL to download from
        target_dir: Directory to save the file
        app_config: Flask app config to check current host
        
    Returns:
        Path to the downloaded file or None if download failed
    """
    try:
        # Make sure target directory exists
        os.makedirs(target_dir, exist_ok=True)
    
        # Handle spaces or special characters in URL
        url = url.strip()
        if ' ' in url:
            logger.warning(f"URL contains spaces, attempting to fix: {url}")
            url = url.replace(' ', '%20')
        
        # Parse URL to get path
        parsed_url = urlparse(url)
        filename = os.path.basename(unquote(parsed_url.path))
        
        logger.info(f"Processing URL {url}, extracted filename: {filename}")
        
        # Check if URL starts with API_HOST
        is_local_file = False
        if app_config and 'API_HOST' in app_config:
            api_host = app_config['API_HOST']
            
            # Direct string comparison - check if URL starts with API_HOST
            if url.startswith(api_host):
                is_local_file = True
                logger.info(f"Detected local file URL that starts with {api_host}")
        
        if is_local_file:
            relative_path = url[len(app_config['API_HOST']):]
            # Decode URL-encoded characters in the path
            relative_path = unquote(relative_path)
            logger.info(f"Local file relative path: {relative_path}")
            
            if relative_path.startswith('/static/uploads/'):
                # Extract the final filename correctly
                filename = os.path.basename(relative_path)
                source_path = os.path.join(
                    app_config.get('STATIC_FOLDER', 'static'),
                    'uploads',
                    filename
                )
                
                # Log debug info about paths
                logger.info(f"Looking for local file at: {source_path}")
                
                # Ensure uploads directory exists
                os.makedirs(os.path.dirname(source_path), exist_ok=True)
                
                if os.path.exists(source_path):
                    target_path = os.path.join(target_dir, filename)
                    # Make sure target directory exists before copying
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    shutil.copy2(source_path, target_path)
                    logger.info(f"Using local file: {source_path} -> {target_path}")
                    return target_path
                else:
                    # Try to find the file with similar name by listing directory contents
                    uploads_dir = os.path.join(app_config.get('STATIC_FOLDER', 'static'), 'uploads')
                    if os.path.exists(uploads_dir):
                        logger.info(f"File not found directly. Searching in directory: {uploads_dir}")
                        # List all files in the directory
                        for file in os.listdir(uploads_dir):
                            # Check if the first part of the filename matches
                            if file.startswith(filename.split('_')[0]):
                                logger.info(f"Found similar file: {file}")
                                matched_source = os.path.join(uploads_dir, file)
                                target_path = os.path.join(target_dir, file)
                                shutil.copy2(matched_source, target_path)
                                logger.info(f"Using found file: {matched_source} -> {target_path}")
                                return target_path
                    
                    logger.warning(f"Local file not found: {source_path}")
            
            elif relative_path.startswith('/images/'):
                # Extract the final filename correctly
                filename = os.path.basename(relative_path)
                source_path = os.path.join(
                    app_config.get('PUBLIC_IMAGES_DIR_ABS', 'public/images'),
                    filename
                )
                
                # Log debug info about paths
                logger.info(f"Looking for local file at: {source_path}")
                
                # Ensure images directory exists
                os.makedirs(os.path.dirname(source_path), exist_ok=True)
                
                if os.path.exists(source_path):
                    target_path = os.path.join(target_dir, filename)
                    # Make sure target directory exists before copying
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    shutil.copy2(source_path, target_path)
                    logger.info(f"Using local file: {source_path} -> {target_path}")
                    return target_path
                else:
                    # Try to find the file with similar name
                    images_dir = app_config.get('PUBLIC_IMAGES_DIR_ABS', 'public/images')
                    if os.path.exists(images_dir):
                        logger.info(f"File not found directly. Searching in directory: {images_dir}")
                        for file in os.listdir(images_dir):
                            if file.startswith(filename.split('_')[0]):
                                logger.info(f"Found similar file: {file}")
                                matched_source = os.path.join(images_dir, file)
                                target_path = os.path.join(target_dir, file)
                                shutil.copy2(matched_source, target_path)
                                logger.info(f"Using found file: {matched_source} -> {target_path}")
                                return target_path
                    
                    logger.warning(f"Local file not found: {source_path}")
        
        # If not local or local file not found, download from URL
        if not filename or filename == '':
            timestamp = int(time.time())
            unique_id = str(uuid.uuid4())[:8]
            extension = os.path.splitext(parsed_url.path)[1].lower()
            if not extension:
                extension = '.jpg'
                
            filename = f"url_file_{timestamp}_{unique_id}{extension}"
        
        file_path = os.path.join(target_dir, filename)
        
        # Ensure parent directory exists before downloading
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        logger.info(f"Downloading file from URL: {url}")
        
        # Use requests session with proper timeout and headers
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        response = session.get(
            url, 
            stream=True, 
            timeout=30, 
            verify=False
        )
        response.raise_for_status()
        
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Successfully downloaded file to: {file_path}")
        return file_path
    
    except Exception as e:
        logger.error(f"Error processing URL {url}: {str(e)}")
        return None

def clean_old_files(app_config, max_age_seconds=3600):
    """
    Clean files that are older than the specified age from public images and uploads directory
    
    Args:
        app_config: Flask app configuration
        max_age_seconds: Maximum age of files in seconds (default: 3600 seconds = 1 hour)
        
    Returns:
        Total number of cleaned items
    """
    total_items = 0
    current_time = time.time()
    
    # Clean old files from public images directory
    public_images_dir = Path(app_config['PUBLIC_IMAGES_DIR_ABS'])
    if public_images_dir.exists():
        for item in public_images_dir.glob("*"):
            try:
                if item.is_file():
                    # Get file modification time
                    file_mtime = item.stat().st_mtime
                    
                    # If file is older than max_age_seconds
                    if current_time - file_mtime > max_age_seconds:
                        item.unlink()
                        total_items += 1
                        logger.info(f"Deleted old file from images: {item.name}")
            except Exception as e:
                logger.warning(f"Error removing old image file {item}: {str(e)}")
    
    # Clean old files from uploads directory
    uploads_dir = Path(app_config.get('STATIC_FOLDER', 'static')) / 'uploads'
    if uploads_dir.exists():
        for item in uploads_dir.glob("*"):
            try:
                if item.is_file():
                    # Get file modification time
                    file_mtime = item.stat().st_mtime
                    
                    # If file is older than max_age_seconds
                    if current_time - file_mtime > max_age_seconds:
                        item.unlink()
                        total_items += 1
                        logger.info(f"Deleted old file from uploads: {item.name}")
            except Exception as e:
                logger.warning(f"Error removing old upload file {item}: {str(e)}")
    
    logger.info(f"Cleaned {total_items} files older than {max_age_seconds/3600:.1f} hours")
    return total_items 