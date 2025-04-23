"""
API routes and endpoint handlers for OMRChecker API
"""
import os
import json
import uuid
import math
import numpy as np
import shutil
import time
import threading
from pathlib import Path
import pandas as pd
import requests
import cv2
import subprocess
import sys

from flask import send_file, send_from_directory, Blueprint, url_for, request, current_app
from flask_restx import Api, Resource
from flask_swagger_ui import get_swaggerui_blueprint
from werkzeug.utils import secure_filename

from src.entry import process_dir
from src.defaults import CONFIG_DEFAULTS
from src.template import Template
from src.logger import logger

from utils.validators import validate_directory_name, clean_nan_values, force_string_conversion
from utils.file_handling import transform_result_format, save_to_public_images, clean_all_folders, download_file_from_url, clean_old_files
from utils.image_processing import process_pdf, validate_image
from api.models import setup_models
from api.parsers import setup_parsers

# Simple in-memory task queue
tasks = {}

class TaskStatusEnum:
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class Task:
    def __init__(self, task_id, directory_name):
        self.task_id = task_id
        self.directory_name = directory_name
        self.status = TaskStatusEnum.PENDING
        self.result = None
        self.error = None
        self.start_time = None
        self.end_time = None
        self.progress = 0
        self.total_files = 0
        self.processed_files = 0
        self.current_stage = "initializing"
        self.stage_progress = 0
        self.stage_details = ""
        self.estimated_remaining_time = None
        self.last_updated = time.time()
        self.results_fetched = 0  # Track how many results have been fetched
        self.temp_file_path = None  # Path to temp file with results
    
    def update_progress(self, stage, progress, details="", increment_processed=0):
        """Update task progress with stage information"""
        self.current_stage = stage
        self.stage_progress = progress
        self.stage_details = details
        self.last_updated = time.time()
        
        if increment_processed > 0:
            self.processed_files += increment_processed
            
        if self.total_files > 0:
            self.progress = min(int((self.processed_files / self.total_files) * 100), 99)
            
        # Rough estimate of remaining time
        if self.progress > 0 and self.start_time:
            elapsed = self.last_updated - self.start_time
            total_estimated = (elapsed / self.progress) * 100
            self.estimated_remaining_time = max(0, total_estimated - elapsed)
            
    def complete(self, result):
        """Mark task as completed with result"""
        self.status = TaskStatusEnum.COMPLETED
        self.result = result
        self.end_time = time.time()
        self.progress = 100
        self.current_stage = "completed"
        self.stage_progress = 100
        self.estimated_remaining_time = 0
        
        # Save results to a temporary file
        try:
            # Use app.config if available, otherwise fallback to default path
            if current_app:
                temp_dir = current_app.config.get('TMP_DIR')
            if not temp_dir:
                temp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tmp")
            
            # Ensure directory exists
            os.makedirs(temp_dir, exist_ok=True)
            
            # Generate a shorter filename to avoid path length issues
            short_task_id = self.task_id[:8]  # Sử dụng 8 ký tự đầu của task_id để rút ngắn tên file
            self.temp_file_path = os.path.join(temp_dir, f"{short_task_id}.json")
            
            # Attempt to store the full results in the temp file
            try:
                # Add metadata to help identify the file
                result_with_meta = result.copy()
                result_with_meta['_meta'] = {
                    'task_id': self.task_id,
                    'saved_at': time.time(),
                    'directory_name': self.directory_name
                }
                
                with open(self.temp_file_path, 'w', encoding='utf-8') as f:
                    json.dump(result_with_meta, f, cls=CustomJSONEncoder)
                
                # Keep only basic info in memory
                result_copy = result.copy()
                if 'results' in result_copy:
                    # Store total results count before clearing
                    total_results = len(result['results'])
                    result_copy['results'] = []  # Clear results array to save memory
                    result_copy['total_results'] = total_results
                self.result = result_copy
                
                logger.info(f"Saved complete task results to temporary file: {self.temp_file_path}")
            except json.JSONEncodeError as json_error:
                logger.error(f"JSON encoding error: {str(json_error)}")
                # Try to save without problematic fields
                if 'results' in result:
                    for i, res in enumerate(result['results']):
                        if 'input_image_url' in res:
                            del res['input_image_url']
                        if 'output_image_url' in res:
                            del res['output_image_url']
                    
                    with open(self.temp_file_path, 'w', encoding='utf-8') as f:
                        json.dump(result, f, cls=CustomJSONEncoder)
                    
                    logger.info(f"Saved simplified results to temporary file after encoding error")
                else:
                    # Keep full results in memory if saving to file fails completely
                    self.result = result
                    logger.error(f"Could not save results to temp file - keeping in memory")
            
        except Exception as e:
            logger.error(f"Failed to save task results to temporary file: {str(e)}")
            # Keep full results in memory if saving to file fails
            self.result = result
        
    def fail(self, error_message):
        """Mark task as failed with error message"""
        self.status = TaskStatusEnum.FAILED
        self.error = error_message
        self.end_time = time.time()
        self.current_stage = "failed"

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float):
            if math.isnan(obj):
                return ""
            if math.isinf(obj):
                return str(obj)
        return super().default(obj)

def setup_routes(app):
    """Set up API routes and blueprints"""
    
    # Set custom JSON encoder to handle NaN and Infinity values
    app.json_encoder = CustomJSONEncoder
    
    # Add CORS headers to all responses
    @app.after_request
    def add_cors_headers(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,X-Requested-With')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        return response
    
    # Handle OPTIONS requests for CORS preflight
    @app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
    @app.route('/<path:path>', methods=['OPTIONS'])
    def handle_options(path):
        return '', 204
    
    # Ensure all required directories exist
    os.makedirs(app.config['INPUTS_DIR_ABS'], exist_ok=True)
    os.makedirs(app.config['OUTPUTS_DIR_ABS'], exist_ok=True)
    os.makedirs(app.config['PROCESSED_DIR'], exist_ok=True)
    os.makedirs(app.config['PUBLIC_IMAGES_DIR_ABS'], exist_ok=True)
    
    # Create tmp directory for temporary task results
    tmp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    app.config['TMP_DIR'] = tmp_dir
    logger.info(f"Ensured all required directories exist, including tmp: {tmp_dir}")
    
    # Create API blueprint
    blueprint = Blueprint('api', __name__, url_prefix='/api')
    
    # Define API with authorization
    authorizations = {
        'apikey': {
            'type': 'apiKey',
            'in': 'header',
            'name': 'Authorization'
        }
    }
    
    api = Api(
        blueprint,
        version='1.0',
        title='OMRChecker API',
        description='API for OMR sheet processing',
        doc='/docs',
        authorizations=authorizations
    )
    
    # Register blueprint with Flask app
    app.register_blueprint(blueprint)
    
    # Set up Swagger UI
    SWAGGER_URL = '/swagger'
    API_URL = '/api/swagger.json'
    
    swaggerui_blueprint = get_swaggerui_blueprint(
        SWAGGER_URL,
        API_URL,
        config={
            'app_name': "OMRChecker API",
            'layout': 'BaseLayout',
            'supportedSubmitMethods': ['get', 'post', 'put', 'delete', 'patch'],
            'jsonEditor': True,
            'validatorUrl': None,
        }
    )
    app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)
    
    # Create API namespace
    ns = api.namespace('', description='OMR operations')
    
    # Set up models and parsers
    models = setup_models(api)
    parsers = setup_parsers()
    
    # Ensure uploads directory exists
    uploads_dir = os.path.join(app.static_folder, 'uploads')
    os.makedirs(uploads_dir, exist_ok=True)
    
    # Define routes
    @app.route('/')
    def index():
        return send_from_directory('static', 'index.html')
    
    @app.route('/static/<path:path>')
    def send_static(path):
        return send_from_directory('static', path)
    
    @app.route('/static/uploads/<path:filename>')
    def serve_uploaded_file(filename):
        return send_from_directory(os.path.join(app.static_folder, 'uploads'), filename)
    
    @app.route('/images/<filename>')
    def serve_public_image(filename):
        return send_from_directory(app.config['PUBLIC_IMAGES_DIR_ABS'], filename)
    
    @app.route('/api/swagger.json')
    def swagger_json():
        schema_json = api.__schema__
        return app.response_class(
            json.dumps(schema_json, cls=CustomJSONEncoder),
            mimetype='application/json'
        )
    
    # Function to process OMR in background
    def process_omr_task(app_config, task_id, args, file_urls):
        task = tasks[task_id]
        task.status = TaskStatusEnum.PROCESSING
        task.start_time = time.time()
        task.update_progress("initializing", 0, "Setting up task")
        
        try:
            with app.app_context():
                directory_name = task.directory_name
                
                clean_before = args['clean_before']
                if isinstance(clean_before, str):
                    clean_before = clean_before.lower() != 'false'
                
                clean_after = args['clean_after']
                if isinstance(clean_after, str):
                    clean_after = clean_after.lower() != 'false'
                
                logger.info(f"Processing OMR for directory: {directory_name} (Task ID: {task_id})")
                
                is_valid, error_message = validate_directory_name(directory_name)
                if not is_valid:
                    task.status = TaskStatusEnum.FAILED
                    task.error = error_message
                    task.end_time = time.time()
                    return
                
                task.update_progress("preparing", 10, "Creating directories")
                # Ensure INPUTS_DIR_ABS exists
                os.makedirs(app_config['INPUTS_DIR_ABS'], exist_ok=True)
                
                input_dir = os.path.join(app_config['INPUTS_DIR_ABS'], directory_name)
                
                if clean_before and os.path.exists(input_dir):
                    try:
                        shutil.rmtree(input_dir)
                    except Exception as e:
                        logger.warning(f"Error cleaning input directory: {str(e)}")
                
                # Create input directory if it doesn't exist
                os.makedirs(input_dir, exist_ok=True)
                
                # Ensure OUTPUTS_DIR_ABS exists
                os.makedirs(app_config['OUTPUTS_DIR_ABS'], exist_ok=True)
                
                output_dir = os.path.join(app_config['OUTPUTS_DIR_ABS'], directory_name)
                
                if clean_before and os.path.exists(output_dir):
                    try:
                        shutil.rmtree(output_dir)
                    except Exception as e:
                        logger.warning(f"Error cleaning output directory: {str(e)}")
                
                # Create output directory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)
                
                try:
                    task.update_progress("setup", 20, "Setting up template and marker files")
                    # Handle template file
                    if args.get('saved_template_path'):
                        template_path = os.path.join(input_dir, 'template.json')
                        shutil.copy2(args['saved_template_path'], template_path)
                        task.update_progress("setup", 25, "Template file processed")
                    elif args.get('template_url'):
                        template_url = args.get('template_url')
                        task.update_progress("setup", 22, f"Downloading template from URL")
                        try:
                            response = requests.get(template_url, timeout=30)
                            if response.status_code != 200:
                                task.fail(f'Failed to download template file from URL: {template_url}, status code: {response.status_code}')
                                return
                            
                            template_path = os.path.join(input_dir, 'template.json')
                            with open(template_path, 'wb') as f:
                                f.write(response.content)
                            task.update_progress("setup", 25, "Template downloaded and saved")
                        except Exception as e:
                            task.fail(f'Error downloading template from URL: {str(e)}')
                            return
                    else:
                        task.fail('No template file provided')
                        return
                    
                    # Handle marker file
                    if args.get('saved_marker_path'):
                        marker_path = os.path.join(input_dir, 'marker.png')
                        shutil.copy2(args['saved_marker_path'], marker_path)
                        task.update_progress("setup", 30, "Marker file processed")
                    elif args.get('marker_url'):
                        marker_url = args.get('marker_url')
                        task.update_progress("setup", 28, f"Downloading marker from URL")
                        try:
                            response = requests.get(marker_url, timeout=30)
                            if response.status_code != 200:
                                task.fail(f'Failed to download marker file from URL: {marker_url}, status code: {response.status_code}')
                                return
                            
                            marker_path = os.path.join(input_dir, 'marker.png')
                            with open(marker_path, 'wb') as f:
                                f.write(response.content)
                            task.update_progress("setup", 30, "Marker downloaded and saved")
                        except Exception as e:
                            task.fail(f'Error downloading marker from URL: {str(e)}')
                            return
                    else:
                        task.fail('No marker file provided')
                        return
                    
                    pdf_files = []
                    regular_image_files = []
                    
                    task.update_progress("processing_files", 35, "Processing input files")
                    # Handle saved image files
                    if args.get('saved_image_files'):
                        saved_files_count = len(args['saved_image_files'])
                        task.update_progress("processing_files", 35, f"Processing {saved_files_count} saved files")
                        for i, img_path in enumerate(args['saved_image_files']):
                            filename = os.path.basename(img_path)
                            dest_path = os.path.join(input_dir, filename)
                            shutil.copy2(img_path, dest_path)
                            
                            file_progress = 35 + (i / saved_files_count * 5)
                            task.update_progress("processing_files", file_progress, f"Processed file {i+1}/{saved_files_count}: {filename}")
                            
                            if img_path.lower().endswith('.pdf'):
                                pdf_files.append(dest_path)
                            else:
                                regular_image_files.append(dest_path)
                    
                    # Process URL files
                    if file_urls:
                        url_count = len(file_urls)
                        task.update_progress("downloading", 40, f"Downloading {url_count} files from URLs")
                        for i, file_url in enumerate(file_urls):
                            
                            # Check if URL ends with space or has encoding issues
                            if file_url.strip() != file_url:
                                logger.warning(f"URL contains leading/trailing spaces, cleaning: '{file_url}'")
                                file_url = file_url.strip()

                            url_progress = 40 + (i / url_count * 10)
                            task.update_progress("downloading", url_progress, f"Downloading file {i+1}/{url_count}")
                            downloaded_path = download_file_from_url(file_url, input_dir, app_config)
                            
                            if downloaded_path:
                                # For PDF files, skip validation since they're not images
                                if downloaded_path.lower().endswith('.pdf'):
                                    logger.info(f"Adding PDF file without image validation: {downloaded_path}")
                                    pdf_files.append(downloaded_path)
                                else:
                                    # Only validate image files
                                    is_valid, error_message = validate_image(downloaded_path)
                                    
                                    if is_valid:
                                        regular_image_files.append(downloaded_path)
                                    else:
                                        logger.warning(f"Skipping invalid image from URL {file_url}: {error_message}")
                            else:
                                logger.warning(f"Failed to download file from URL: {file_url}")
                    
                    image_paths = regular_image_files.copy()
                    
                    # Validate regular image files before processing
                    valid_image_paths = []
                    if image_paths:
                        task.update_progress("validating", 50, f"Validating {len(image_paths)} image files")
                        for i, img_path in enumerate(image_paths):
                            is_valid, error_message = validate_image(img_path)
                            
                            if is_valid:
                                valid_image_paths.append(img_path)
                            else:
                                logger.warning(f"Skipping invalid image: {error_message}")
                            
                            validation_progress = 50 + (i / len(image_paths) * 5)
                            task.update_progress("validating", validation_progress, f"Validated file {i+1}/{len(image_paths)}")
                    
                    # Update image_paths with only valid images
                    image_paths = valid_image_paths
                    
                    # Update task with total files count
                    task.total_files = len(image_paths) + len(pdf_files)
                    task.update_progress("processing", 55, f"Processing {task.total_files} files in total")
                    
                    # Process PDFs if any
                    if pdf_files:
                        pdf_count = len(pdf_files)
                        logger.info(f"Processing {pdf_count} PDF files in batch mode")
                        task.update_progress("pdf_processing", 60, f"Extracting images from {pdf_count} PDF files")
                        pdf_start_time = time.time()
                        
                        try:
                            from utils.image_processing import process_pdf_batch
                            from utils.batch_config import get_batch_profile
                            
                            batch_profile = get_batch_profile(pdf_count)
                            
                            high_dpi = 600
                            high_quality = 100
                            
                            pdf_results = process_pdf_batch(
                                pdf_files, 
                                input_dir,
                                dpi=high_dpi,
                                quality=high_quality
                            )
                            
                            extracted_images = 0
                            for pdf_path, paths in pdf_results.items():
                                image_paths.extend(paths)
                                extracted_images += len(paths)
                                
                                try:
                                    os.remove(pdf_path)
                                except Exception as rm_error:
                                    logger.warning(f"Could not remove PDF file {pdf_path}: {str(rm_error)}")
                            
                            pdf_total_time = time.time() - pdf_start_time
                            logger.info(f"Batch PDF processing completed in {pdf_total_time:.2f} seconds for {pdf_count} PDFs")
                            task.update_progress("pdf_processing", 70, f"Extracted {extracted_images} images from {pdf_count} PDF files")
                            
                        except Exception as batch_error:
                            logger.error(f"Error in batch PDF processing: {str(batch_error)}")
                            logger.info("Falling back to individual PDF processing")
                            task.update_progress("pdf_processing", 60, f"Batch processing failed, falling back to individual processing")
                            
                            from utils.image_processing import process_pdf
                            
                            for i, pdf_path in enumerate(pdf_files):
                                try:
                                    individual_progress = 60 + (i / pdf_count * 10)
                                    task.update_progress("pdf_processing", individual_progress, f"Processing PDF {i+1}/{pdf_count}")
                                    
                                    new_image_paths = process_pdf(
                                        pdf_path, 
                                        input_dir,
                                        dpi=600,
                                        quality=100,
                                        max_workers=12
                                    )
                                    image_paths.extend(new_image_paths)
                                    task.update_progress("pdf_processing", individual_progress + 5, 
                                                         f"Extracted {len(new_image_paths)} images from PDF {i+1}/{pdf_count}")
                                    
                                    try:
                                        os.remove(pdf_path)
                                    except Exception as rm_error:
                                        logger.warning(f"Could not remove PDF file {pdf_path}: {str(rm_error)}")
                                    
                                except Exception as pdf_error:
                                    logger.error(f"Error processing PDF file {pdf_path}: {str(pdf_error)}")
                    
                    if not image_paths:
                        task.fail('No valid images to process')
                        return
                                    
                    task.update_progress("omr_processing", 75, f"Starting OMR processing for {len(image_paths)} images")                
                    api_args = {
                        'input_paths': [input_dir],
                        'output_dir': app_config['OUTPUTS_DIR_ABS'],
                        'autoAlign': False,
                        'setLayout': False,
                        'debug': True,
                    }
                    
                    tuning_config = CONFIG_DEFAULTS
                    
                    if pdf_files:
                        from utils.batch_config import get_batch_profile
                        batch_profile = get_batch_profile(len(pdf_files))
                        
                        tuning_config.dimensions.processing_width = batch_profile["processing_width"]
                        tuning_config.outputs.save_image_level = batch_profile["save_image_level"]
                        tuning_config.outputs.show_image_level = batch_profile["show_image_level"]
                    else:
                        tuning_config.dimensions.processing_width = 1200
                        tuning_config.outputs.save_image_level = 0
                        tuning_config.outputs.show_image_level = 0
                    
                    template = Template(Path(template_path), tuning_config)
                    
                    root_dir = Path(app_config['INPUTS_DIR_ABS'])
                    curr_dir = Path(input_dir)
                    
                    omr_start_time = time.time()
                    
                    task.update_progress("omr_scanning", 80, f"Scanning OMR sheets using CLI")
                    
                    output_path = os.path.join(app_config['OUTPUTS_DIR_ABS'], directory_name)
                    
                    cmd = [
                        sys.executable,
                        "main.py", 
                        "-i", str(curr_dir),
                        "-o", output_path,
                        "--debug"
                    ]
                    
                    logger.info(f"Executing command: {' '.join(cmd)}")
                    
                    try:
                        process = subprocess.Popen(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            universal_newlines=True
                        )
                        
                        task.update_progress("omr_preprocessing", 81, f"Pre-processing images for OMR recognition")
                        task.update_progress("marker_detection", 82, f"Detecting markers in images")
                        task.update_progress("image_alignment", 83, f"Aligning images with template")
                        task.update_progress("bubble_detection", 85, f"Detecting and analyzing bubbles")
                        task.update_progress("scoring", 88, f"Calculating scores from detected marks")
                        
                        stdout, stderr = process.communicate()
                        
                        if process.returncode != 0:
                            logger.error(f"OMR processing failed with error: {stderr}")
                            task.fail(f"OMR processing command failed: {stderr}")
                            return
                            
                        logger.info(f"OMR processing completed via CLI command")
                        
                        results = {
                            "status": "success",
                            "command": " ".join(cmd),
                            "output": stdout
                        }
                    except Exception as cmd_error:
                        logger.error(f"Error executing OMR command: {str(cmd_error)}")
                        task.fail(f"Error executing OMR command: {str(cmd_error)}")
                        return
                    
                    task.update_progress("results_processing", 90, f"Processing results")
                    
                    output_dir_path = Path(output_dir)
                    
                    all_csv_files = list(output_dir_path.glob('**/*.csv'))
                    
                    results_file = list(output_dir_path.glob('**/CheckedOMRs/*.csv'))
                    
                    task.update_progress("locating_results", 91, f"Locating result files")
                    
                    if not results_file:
                        results_file = list(output_dir_path.glob('**/Results/*.csv'))
                    
                    if not results_file:
                        results_file = list(output_dir_path.glob('**/Results_*.csv'))
                    
                    if not results_file:
                        non_error_csv = [f for f in all_csv_files if 'ErrorFiles.csv' not in str(f)]
                        if non_error_csv:
                            results_file = [non_error_csv[0]]
                        else:
                            results_file = all_csv_files
                    
                    task.update_progress("extracting_data", 92, f"Extracting data from result files")
                    
                    if not results_file:
                        task.fail('Processing failed - no results generated')
                        return
                    
                    csv_file_path = results_file[0]
                    
                    task.update_progress("preparing_results", 93, f"Reading results from {csv_file_path.name}")
                    df = pd.read_csv(csv_file_path, dtype={'studentId': str, 'code': str})
                    
                    df = force_string_conversion(df, ['studentId', 'code'])
                    
                    df = df.replace([np.inf, -np.inf], 'Infinity')
                    df = df.fillna("")
                    
                    results_data = df.to_dict(orient='records')
                    
                    result_id = str(uuid.uuid4())
                    result_dir = Path(app_config['PROCESSED_DIR']) / result_id
                    os.makedirs(result_dir, exist_ok=True)
                    
                    task.update_progress("saving_results", 97, f"Saving result files")
                    copied_files = 0
                    skipped_files = 0
                    
                    for file in output_dir_path.glob('**/*'):
                        if file.is_file():
                            try:
                                rel_path = file.relative_to(output_dir_path)
                                target_path = result_dir / rel_path
                                
                                source_path_str = str(file)
                                target_path_str = str(target_path)
                                
                                if os.name == 'nt' and (len(source_path_str) > 250 or len(target_path_str) > 250):
                                    if not source_path_str.startswith("\\\\?\\"):
                                        source_path_str = "\\\\?\\" + os.path.abspath(source_path_str)
                                    if not target_path_str.startswith("\\\\?\\"):
                                        target_path_str = "\\\\?\\" + os.path.abspath(target_path_str)
                                
                                os.makedirs(os.path.dirname(target_path_str), exist_ok=True)
                                
                                try:
                                    shutil.copy2(source_path_str, target_path_str)
                                    copied_files += 1
                                except (OSError, IOError) as copy_error:
                                    try:
                                        with open(source_path_str, 'rb') as src_file:
                                            with open(target_path_str, 'wb') as dst_file:
                                                dst_file.write(src_file.read())
                                        copied_files += 1
                                        logger.info(f"Successfully copied file using manual read/write: {rel_path}")
                                    except Exception as manual_error:
                                        logger.warning(f"Error copying file using all methods {rel_path}: {str(manual_error)}")
                                        skipped_files += 1
                            except Exception as path_error:
                                logger.warning(f"Error with file path {file}: {str(path_error)}")
                                skipped_files += 1
                    
                    logger.info(f"Copied {copied_files} files, skipped {skipped_files} files")
                    task.update_progress("processing_images", 98, f"Processing result images, copied {copied_files} files")
                    
                    total_result_images = len(results_data)
                    task.update_progress("counting_images", 98.2, f"Found {total_result_images} result images to process")
                    
                    batch_size = min(10, max(1, total_result_images // 5))
                    total_batches = math.ceil(total_result_images / batch_size)
                    task.update_progress("image_batching", 98.3, f"Processing images in {total_batches} batches")
                    
                    clean_results = []
                    for batch_idx, batch_start in enumerate(range(0, total_result_images, batch_size)):
                        batch_end = min(batch_start + batch_size, total_result_images)
                        batch_results = results_data[batch_start:batch_end]
                        
                        batch_progress = 98.3 + (batch_idx / total_batches * 0.6)
                        task.update_progress("batch_processing", batch_progress, 
                                            f"Processing image batch {batch_idx+1}/{total_batches} ({batch_start+1}-{batch_end}/{total_result_images})")
                        
                        for result in batch_results:
                            clean_result = clean_nan_values(result)
                            transformed_result = transform_result_format(clean_result)
                            
                            input_image_path = None
                            output_image_path = None
                            
                            if 'input_path' in clean_result:
                                input_image_path = clean_result['input_path']
                                if input_image_path and not os.path.exists(input_image_path):
                                    logger.warning(f"Input image path does not exist: {input_image_path}")
                                    input_image_path = None
                            elif 'file_id' in clean_result:
                                file_id = clean_result['file_id']
                                for img_path in image_paths:
                                    if os.path.basename(img_path) == file_id:
                                        input_image_path = img_path
                                        break
                                
                                if not input_image_path:
                                    logger.warning(f"Could not find image for file_id: {file_id}")
                            
                            if 'output_path' in clean_result:
                                output_image_path = clean_result['output_path']
                                if output_image_path and not os.path.exists(output_image_path):
                                    logger.warning(f"Output image path does not exist: {output_image_path}")
                                    output_image_path = None
                            
                            public_input_image = None
                            public_output_image = None
                            
                            os.makedirs(app_config['PUBLIC_IMAGES_DIR_ABS'], exist_ok=True)
                            
                            if input_image_path and os.path.exists(input_image_path):
                                try:
                                    if os.name == 'nt' and len(input_image_path) > 250:
                                        input_image_path_unc = "\\\\?\\" + os.path.abspath(input_image_path)
                                    else:
                                        input_image_path_unc = input_image_path
                                        
                                    public_input_image = save_to_public_images(
                                        input_image_path_unc, 
                                        "input", 
                                        app_config['API_HOST'], 
                                        app_config['PUBLIC_IMAGES_DIR_ABS']
                                    )
                                    if public_input_image:
                                        transformed_result['input_image_url'] = public_input_image
                                except Exception as e:
                                    logger.warning(f"Error saving input image to public folder: {str(e)}")
                            
                            if output_image_path and os.path.exists(output_image_path):
                                try:
                                    if os.name == 'nt' and len(output_image_path) > 250:
                                        output_image_path_unc = "\\\\?\\" + os.path.abspath(output_image_path)
                                    else:
                                        output_image_path_unc = output_image_path
                                        
                                    public_output_image = save_to_public_images(
                                        output_image_path_unc, 
                                        "output", 
                                        app_config['API_HOST'], 
                                        app_config['PUBLIC_IMAGES_DIR_ABS']
                                    )
                                    if public_output_image:
                                        transformed_result['output_image_url'] = public_output_image
                                except Exception as e:
                                    logger.warning(f"Error saving output image to public folder: {str(e)}")
                            
                            clean_results.append(transformed_result)
                    
                    end_time = time.time()
                    total_processing_time = end_time - omr_start_time
                    logger.info(f"Total processing completed in {total_processing_time:.2f} seconds")
                    
                    task.update_progress("finalizing", 99, f"Finalizing results")
                    
                    task.update_progress("calculating_stats", 99.2, f"Calculating result statistics")
                    
                    total_processed = len(clean_results)
                    results_with_images = sum(1 for r in clean_results if 'input_image_url' in r or 'output_image_url' in r)
                    
                    task.update_progress("formatting_results", 99.4, f"Formatting {total_processed} results ({results_with_images} with images)")
                    
                    task.update_progress("preparing_response", 99.6, f"Preparing response data")
                    
                    response_data = {
                        'message': 'OMR processing completed successfully',
                        'result_id': result_id,
                        'task_id': task_id,
                        'input_dir': str(input_dir),
                        'output_dir': str(output_dir),
                        'csv_file': str(csv_file_path.name),
                        'timing': {
                            'total_processing': round(total_processing_time, 2)
                        },
                        'statistics': {
                            'total_processed': total_processed,
                            'with_images': results_with_images,
                            'processing_time_seconds': round(total_processing_time, 2)
                        },
                        'results': clean_results
                    }
                    
                    task.update_progress("wrapping_up", 99.8, f"Wrapping up processing after {round(total_processing_time, 2)} seconds")
                    
                    task.complete(response_data)
                    
                    if clean_after:
                        try:
                            if os.path.exists(input_dir):
                                shutil.rmtree(input_dir)
                            
                            if os.path.exists(output_dir):
                                shutil.rmtree(output_dir)
                        except Exception as e:
                            logger.warning(f"Error cleaning directories after processing: {str(e)}")
                    
                    temp_dir = os.path.join(app_config['INPUTS_DIR_ABS'], f"temp_{task_id}")
                    if os.path.exists(temp_dir):
                        try:
                            shutil.rmtree(temp_dir)
                        except Exception as e:
                            logger.warning(f"Error cleaning temp directory: {str(e)}")
                    
                except Exception as e:
                    logger.error(f"Error processing OMR: {str(e)}")
                    task.fail(f'Error processing OMR: {str(e)}')
                    
                    temp_dir = os.path.join(app_config['INPUTS_DIR_ABS'], f"temp_{task_id}")
                    if os.path.exists(temp_dir):
                        try:
                            shutil.rmtree(temp_dir)
                        except Exception as clean_err:
                            logger.warning(f"Error cleaning temp directory after error: {str(clean_err)}")

        except Exception as e:
            logger.error(f"Unexpected error in task processing: {str(e)}")
            task.fail(f'Unexpected error: {str(e)}')
    
    @ns.route('/process-omr')
    @ns.expect(parsers['upload_parser'])
    class ProcessOMR(Resource):
        @ns.doc('process_omr', 
                responses={
                    200: 'Task created successfully',
                    400: 'Validation Error',
                    500: 'Processing Error'
                })
        def post(self):
            """Create a task to process multiple OMR sheets with the provided template"""
            try:
                args = parsers['upload_parser'].parse_args()
                
                directory_name = args['directory_name']
                
                logger.info(f"Creating task for OMR processing, directory: {directory_name}")
                
                is_valid, error_message = validate_directory_name(directory_name)
                if not is_valid:
                    return {"error": error_message}, 400
                
                image_files = args['image_files'] or []
                file_urls = args['file_urls'] or []
                
                if not image_files and not file_urls:
                    return {'error': 'At least one image file or file URL must be provided'}, 400
                
                # Generate a task ID
                task_id = str(uuid.uuid4())
                
                # Create a new task
                task = Task(task_id, directory_name)
                tasks[task_id] = task
                
                # Create a temporary directory for this task
                temp_dir = os.path.join(app.config['INPUTS_DIR_ABS'], f"temp_{task_id}")
                os.makedirs(temp_dir, exist_ok=True)
                
                # Save uploaded files to temp directory
                saved_files = []
                if image_files:
                    for file in image_files:
                        temp_path = os.path.join(temp_dir, secure_filename(file.filename))
                        file.save(temp_path)
                        saved_files.append(temp_path)
                
                # Save template and marker files if present
                template_path = None
                if args.get('template_file'):
                    template_file = args.get('template_file')
                    template_path = os.path.join(temp_dir, 'template.json')
                    template_file.save(template_path)
                
                marker_path = None
                if args.get('marker_file'):
                    marker_file = args.get('marker_file')
                    marker_path = os.path.join(temp_dir, 'marker.png')
                    marker_file.save(marker_path)
                
                # Create a copy of args without file objects
                args_copy = args.copy()
                args_copy['image_files'] = None
                args_copy['template_file'] = None
                args_copy['marker_file'] = None
                
                # Add paths to saved files
                args_copy['saved_image_files'] = saved_files
                args_copy['saved_template_path'] = template_path
                args_copy['saved_marker_path'] = marker_path
                
                # Start a new thread to process the task
                thread = threading.Thread(
                    target=process_omr_task,
                    args=(app.config.copy(), task_id, args_copy, file_urls)
                )
                thread.daemon = True
                thread.start()
                
                return {
                    'message': 'OMR processing task created successfully',
                    'task_id': task_id,
                    'status': TaskStatusEnum.PENDING,
                    'directory_name': directory_name
                }, 200
                
            except Exception as e:
                logger.error(f"Error creating OMR processing task: {str(e)}")
                return {'error': f'Error creating OMR processing task: {str(e)}'}, 500

        @ns.doc('process_omr_get',
                responses={
                    400: 'Missing Parameters',
                    404: 'Directory Not Found',
                    500: 'Processing Error'
                })
        def get(self):
            """Return status information for a directory without processing files (status check only)"""
            args = parsers['get_params_parser'].parse_args()
            
            directory_name = args['directory_name']
            
            clean_before = args['clean_before']
            if isinstance(clean_before, str):
                clean_before = clean_before.lower() != 'false'
            
            clean_after = args['clean_after']
            if isinstance(clean_after, str):
                clean_after = clean_after.lower() != 'false'
            
            logger.info(f"Checking status for directory: {directory_name}")
            
            is_valid, error_message = validate_directory_name(directory_name)
            if not is_valid:
                return {'error': error_message}, 400
            
            input_dir = os.path.join(app.config['INPUTS_DIR_ABS'], directory_name)
            output_dir = os.path.join(app.config['OUTPUTS_DIR_ABS'], directory_name)
            
            input_exists = os.path.exists(input_dir)
            output_exists = os.path.exists(output_dir)

            input_files = []
            output_files = []
            
            if input_exists:
                input_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
                
            if output_exists:
                output_files = []
                for root, _, files in os.walk(output_dir):
                    for file in files:
                        rel_path = os.path.relpath(os.path.join(root, file), output_dir)
                        output_files.append(rel_path)
            
            return {
                'message': 'Directory information retrieved successfully',
                'directory_name': directory_name,
                'input_directory': {
                    'exists': input_exists,
                    'path': str(input_dir),
                    'files': input_files
                },
                'output_directory': {
                    'exists': output_exists,
                    'path': str(output_dir),
                    'files': output_files
                },
                'clean_before': clean_before,
                'clean_after': clean_after
            }, 200

    @ns.route('/tasks/<string:task_id>')
    @ns.param('task_id', 'The unique identifier for the task')
    class TaskStatus(Resource):
        @ns.doc('get_task_status', 
                responses={
                    200: 'Success', 
                    404: 'Task not found'
                })
        def get(self, task_id):
            """Get status of a specific OMR processing task and results with pagination"""
            # Parse parameters for pagination
            page = request.args.get('page', default=1, type=int)
            page_size = request.args.get('page_size', default=10, type=int)
            
            # Validate page and page_size
            if page < 1:
                page = 1
            if page_size < 1:
                page_size = 10
            if page_size > 100:
                page_size = 100
                
            if task_id not in tasks:
                return {'error': 'Task not found'}, 404
            
            task = tasks[task_id]
            
            response = {
                'task_id': task.task_id,
                'directory_name': task.directory_name,
                'status': task.status,
                'progress': task.progress,
                'total_files': task.total_files,
                'processed_files': task.processed_files,
                'created_at': task.start_time or time.time(),
                'current_stage': task.current_stage,
                'stage_progress': task.stage_progress,
                'stage_details': task.stage_details,
                'last_updated': task.last_updated,
                'page': page,
                'page_size': page_size
            }
            
            if task.estimated_remaining_time is not None:
                response['estimated_remaining_seconds'] = round(task.estimated_remaining_time)
                
                # Add human-readable estimate
                minutes, seconds = divmod(round(task.estimated_remaining_time), 60)
                if minutes > 0:
                    response['estimated_remaining'] = f"{minutes}m {seconds}s"
                else:
                    response['estimated_remaining'] = f"{seconds}s"
            
            # For completed tasks, handle results with pagination
            if task.status == TaskStatusEnum.COMPLETED:
                # Remove full results array from response
                if 'result' in response:
                    del response['result']
                
                # Get basic task result info (without the large results array)
                response.update(task.result)
                
                # Calculate total pages
                total_results = task.result.get('total_results', 0)
                total_pages = math.ceil(total_results / page_size) if total_results > 0 else 0
                response['total_pages'] = total_pages
                
                # If temp file exists, read paginated results from it
                if task.temp_file_path and os.path.exists(task.temp_file_path):
                    try:
                        with open(task.temp_file_path, 'r', encoding='utf-8') as f:
                            full_result = json.load(f)
                            
                        if 'results' in full_result:
                            # Calculate pagination
                            start_idx = (page - 1) * page_size
                            end_idx = start_idx + page_size
                            
                            # Get paginated results
                            if start_idx < len(full_result['results']):
                                response['results'] = full_result['results'][start_idx:end_idx]
                                
                                # Update how many results have been fetched
                                task.results_fetched = max(task.results_fetched, end_idx)
                                
                                # Add pagination info to response
                                response['pagination'] = {
                                    'page': page,
                                    'page_size': page_size,
                                    'total_results': total_results,
                                    'total_pages': total_pages,
                                    'has_next': page < total_pages,
                                    'has_prev': page > 1
                                }
                            else:
                                response['results'] = []
                                response['message'] = f"Page {page} exceeds available results"
                            
                            # Only delete the task and temp file if we've fetched all results and we're on the last page
                            if task.results_fetched >= total_results and page >= total_pages:
                                logger.info(f"All results fetched for task {task_id}, cleaning up")
                                
                                # Delete the temp file
                                try:
                                    os.remove(task.temp_file_path)
                                    logger.info(f"Deleted temp file: {task.temp_file_path}")
                                except Exception as e:
                                    logger.warning(f"Failed to delete temp file: {str(e)}")
                                
                                # Delete the task from memory
                                del tasks[task_id]
                                response['message'] = 'All results retrieved, task data removed from memory'
                    except Exception as e:
                        logger.error(f"Error reading task results from temp file: {str(e)}")
                        response['error'] = f"Error reading results: {str(e)}"
                
                response['completed_at'] = task.end_time
                response['processing_time'] = round(task.end_time - task.start_time, 2) if task.start_time else None
            
            if task.status == TaskStatusEnum.FAILED:
                response['error'] = task.error
                response['failed_at'] = task.end_time
                
                # Delete failed tasks immediately
                del tasks[task_id]
                response['message'] = 'Task data removed from memory due to failure'
            
            return response, 200

    @ns.route('/results/<string:result_id>')
    @ns.param('result_id', 'The unique identifier for the result set')
    class Results(Resource):
        @ns.doc('get_results', 
                responses={
                    200: 'Success', 
                    404: 'Result not found'
                })
        def get(self, result_id):
            """Get results for a specific processed OMR"""
            result_dir = Path(app.config['PROCESSED_DIR']) / result_id
            
            if not result_dir.exists():
                return {'error': 'Result not found'}, 404
            
            csv_files = list(result_dir.glob('**/CheckedOMRs/*.csv'))
            
            if not csv_files:
                csv_files = list(result_dir.glob('**/Results/*.csv')) 
            
            if not csv_files:
                csv_files = list(result_dir.glob('**/Results_*.csv'))
            
            if not csv_files:
                all_csv = list(result_dir.glob('**/*.csv'))
                non_error_csv = [f for f in all_csv if 'ErrorFiles.csv' not in str(f)]
                if non_error_csv:
                    csv_files = [non_error_csv[0]]
                else:
                    csv_files = all_csv
                    
            if not csv_files:
                return {'error': 'No results found'}, 404
            
            df = pd.read_csv(csv_files[0], dtype={'studentId': str, 'code': str})
            
            df = force_string_conversion(df, ['studentId', 'code'])
            
            df = df.replace([np.inf, -np.inf], 'Infinity')
            df = df.fillna("")
            
            results_data = df.to_dict(orient='records')
            
            clean_results = []
            for result in results_data:
                clean_result = clean_nan_values(result)
                transformed_result = transform_result_format(clean_result)
                clean_results.append(transformed_result)
            
            return {
                'result_id': result_id,
                'csv_file': csv_files[0].name,
                'results': clean_results
            }, 200

    @ns.route('/download/<string:result_id>/<path:filename>')
    @ns.param('result_id', 'The unique identifier for the result set')
    @ns.param('filename', 'The name of the file to download')
    class Download(Resource):
        @ns.doc('download_file',
                responses={
                    200: 'Success',
                    404: 'File not found'
                })
        def get(self, result_id, filename):
            """Download a file from the results"""
            if '..' in filename or filename.startswith('/'):
                return {'error': 'Invalid filename'}, 400
                
            result_dir = Path(app.config['PROCESSED_DIR']) / result_id
            
            if not result_dir.exists():
                return {'error': 'Result not found'}, 404
            
            if filename.lower() == 'results_11am.csv' or filename.lower() == 'results.csv':
                results_file = list(result_dir.glob('**/Results_11AM.csv'))
                if results_file:
                    file_path = results_file[0]
                else:
                    results_file = list(result_dir.glob('**/*.csv'))
                    if not results_file:
                        return {'error': 'No CSV results found'}, 404
                    file_path = results_file[0]
            else:
                file_path = result_dir / filename
                
            if not file_path.exists() or not file_path.is_file():
                return {'error': 'File not found'}, 404
            
            try:
                if not str(file_path.resolve()).startswith(str(result_dir.resolve())):
                    return {'error': 'Invalid file path'}, 403
            except (ValueError, RuntimeError):
                return {'error': 'Invalid file path'}, 403
                
            return send_file(
                file_path,
                as_attachment=True,
                download_name=file_path.name
            )

    @ns.route('/health')
    class Health(Resource):
        @ns.doc('health_check',
                responses={
                    200: 'API is healthy'
                })
        def get(self):
            """Health check endpoint"""
            return {'status': 'healthy'}, 200

    @ns.route('/clean-folders')
    class CleanFolders(Resource):
        @ns.doc('clean_folders',
                responses={
                    200: 'Cleaned successfully',
                    500: 'Cleaning error'
                })
        def post(self):
            """Delete contents of all directories: inputs, outputs, images"""
            try:
                total_items = clean_all_folders(app.config)
                return {
                    'status': 'success',
                    'message': f'Deleted {total_items} items from all directories'
                }, 200
            except Exception as e:
                logger.error(f"Error during manual cleaning: {str(e)}")
                return {'error': f'Error cleaning directories: {str(e)}'}, 500

    @ns.route('/upload-file')
    @ns.expect(parsers['single_file_parser'])
    class UploadSingleFile(Resource):
        @ns.doc('upload_single_file',
                responses={
                    200: 'Success',
                    400: 'Bad Request',
                    413: 'Entity Too Large',
                    500: 'Error'
                })
        def post(self):
            """Upload a single file and return its URL"""
            try:
                args = parsers['single_file_parser'].parse_args()
                file = args['file']
                
                if not file or file.filename == '':
                    return {'error': 'No file selected'}, 400
                
                # Ensure uploads directory exists (double-check)
                uploads_dir = os.path.join(app.static_folder, 'uploads')
                os.makedirs(uploads_dir, exist_ok=True)
                
                filename = secure_filename(file.filename)
                unique_filename = f"{uuid.uuid4().hex}_{filename}"
                file_path = os.path.join(uploads_dir, unique_filename)
                
                # Ensure parent directory exists
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                file.save(file_path)
                
                file_url = url_for('serve_uploaded_file', filename=unique_filename, _external=True)
                
                return {
                    'filename': filename,
                    'url': file_url,
                    'size': os.path.getsize(file_path)
                }, 200
                
            except Exception as e:
                logger.error(f"Error uploading file: {str(e)}")
                return {'error': str(e)}, 500

    @ns.route('/clean-old-files')
    class CleanOldFiles(Resource):
        @ns.doc('clean_old_files',
                responses={
                    200: 'Cleaned successfully',
                    500: 'Cleaning error'
                })
        def post(self):
            """Delete files older than 1 hour from images and uploads directories"""
            try:
                total_items = clean_old_files(app.config)
                return {
                    'status': 'success',
                    'message': f'Deleted {total_items} files older than 1 hour'
                }, 200
            except Exception as e:
                logger.error(f"Error during manual cleaning of old files: {str(e)}")
                return {'error': f'Error cleaning old files: {str(e)}'}, 500

    @ns.route('/process-batch')
    @ns.expect(parsers['upload_parser'])
    class ProcessBatch(Resource):
        @ns.doc('process_batch', 
                responses={
                    200: 'Success',
                    400: 'Validation Error',
                    500: 'Processing Error'
                })
        def post(self):
            """Process large batches of files by splitting them into subfolders"""
            try:
                args = parsers['upload_parser'].parse_args()
                directory_name = args['directory_name']
                clean_before = args['clean_before']
                clean_after = args['clean_after']
                batch_size = args['batch_size'] or 50  # Default to 50 if not provided
                image_files = args['image_files'] or []
                file_urls = args['file_urls'] or []
                template_file = args.get('template_file')
                marker_file = args.get('marker_file')
                template_url = args.get('template_url')
                marker_url = args.get('marker_url')
                
                logger.info(f"Processing batch with directory: {directory_name}, batch size: {batch_size}")
                logger.info(f"Template URL: {template_url}, Marker URL: {marker_url}")
                
                # Validate batch size
                if batch_size < 1 or batch_size > 200:
                    return {'error': 'Batch size must be between 1 and 200'}, 400
                
                # Check if any image files or URLs were provided
                if not image_files and not file_urls:
                    return {'error': 'No image files or URLs provided'}, 400
                
                # Validate that we have template and marker files or URLs
                if not ((template_file or template_url) and (marker_file or marker_url)):
                    return {'error': 'Both template and marker files/URLs are required'}, 400
                
                # Validate directory name
                is_valid, error_message = validate_directory_name(directory_name)
                if not is_valid:
                    return {"error": error_message}, 400
                
                # Create paths for all the required directories
                input_dir = os.path.join(app.config['INPUTS_DIR_ABS'], directory_name)
                output_dir = os.path.join(app.config['OUTPUTS_DIR_ABS'], directory_name)
                
                # Clean directories if needed
                if clean_before:
                    for directory in [input_dir, output_dir]:
                        if os.path.exists(directory):
                            shutil.rmtree(directory)
                
                # Create directories
                for directory in [input_dir, output_dir]:
                    os.makedirs(directory, exist_ok=True)
                
                # Save template file and marker file
                template_path = os.path.join(input_dir, 'template.json')
                marker_path = os.path.join(input_dir, 'marker.png')
                
                # Handle template file (either upload or download from URL)
                if template_file:
                    if not template_file.filename.endswith('.json'):
                        return {'error': 'Template file must be a JSON file'}, 400
                    template_file.save(template_path)
                elif template_url:
                    try:
                        response = requests.get(template_url, timeout=30)
                        if response.status_code != 200:
                            return {'error': f'Failed to download template file from URL: {template_url}, status code: {response.status_code}'}, 400
                        with open(template_path, 'wb') as f:
                            f.write(response.content)
                    except Exception as e:
                        return {'error': f'Error downloading template from URL: {str(e)}'}, 400
                else:
                    return {'error': 'Either template_file or template_url must be provided'}, 400
                
                # Validate template file format
                try:
                    with open(template_path, 'r') as f:
                        json.load(f)
                except json.JSONDecodeError:
                    return {'error': 'Invalid template file format (not valid JSON)'}, 400
                
                # Handle marker file (either upload or download from URL)
                if marker_file:
                    if not any(marker_file.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
                        return {'error': f'Marker file {marker_file.filename} must be a PNG, JPG, or JPEG file'}, 400
                    marker_file.save(marker_path)
                elif marker_url:
                    try:
                        response = requests.get(marker_url, timeout=30)
                        if response.status_code != 200:
                            return {'error': f'Failed to download marker file from URL: {marker_url}, status code: {response.status_code}'}, 400
                        with open(marker_path, 'wb') as f:
                            f.write(response.content)
                    except Exception as e:
                        return {'error': f'Error downloading marker from URL: {str(e)}'}, 400
                else:
                    return {'error': 'Either marker_file or marker_url must be provided'}, 400
                
                # Validate marker file format
                try:
                    marker_img = cv2.imread(marker_path)
                    if marker_img is None:
                        return {'error': 'Invalid marker file format (not a valid image)'}, 400
                except Exception as e:
                    return {'error': f'Error reading marker file: {str(e)}'}, 400
                
                # Process uploaded files and URLs
                image_paths = []
                pdf_files = []
                
                # Save uploaded files
                for image_file in image_files:
                    if image_file.filename.lower().endswith('.pdf'):
                        pdf_path = os.path.join(input_dir, image_file.filename)
                        image_file.save(pdf_path)
                        pdf_files.append(pdf_path)
                    else:
                        image_path = os.path.join(input_dir, image_file.filename)
                        image_file.save(image_path)
                        image_paths.append(image_path)
                
                # Download files from URLs
                for file_url in file_urls:
                    if file_url.strip() != file_url:
                        logger.warning(f"URL contains leading/trailing spaces, cleaning: '{file_url}'")
                        file_url = file_url.strip()
                    
                    downloaded_path = download_file_from_url(file_url, input_dir, app.config)
                    
                    if downloaded_path:
                        if downloaded_path.lower().endswith('.pdf'):
                            pdf_files.append(downloaded_path)
                        else:
                            is_valid, error_message = validate_image(downloaded_path)
                            if is_valid:
                                image_paths.append(downloaded_path)
                            else:
                                logger.warning(f"Skipping invalid image from URL {file_url}: {error_message}")
                    else:
                        logger.warning(f"Failed to download file from URL: {file_url}")
                
                # Process PDFs if any
                if pdf_files:
                    logger.info(f"Processing {len(pdf_files)} PDF files in batch mode")
                    pdf_start_time = time.time()
                    
                    try:
                        from utils.image_processing import process_pdf_batch
                        from utils.batch_config import get_batch_profile
                        
                        batch_profile = get_batch_profile(len(pdf_files))
                        
                        # Tăng DPI và quality lên cao nhất cho độ nét tốt nhất
                        high_dpi = 300  # Giá trị DPI cao hơn cho độ nét tốt
                        high_quality = 95  # Giá trị quality cao hơn (max 100)
                        
                        pdf_results = process_pdf_batch(
                            pdf_files, 
                            input_dir,
                            dpi=high_dpi,
                            quality=high_quality
                        )
                        
                        extracted_images = 0
                        for pdf_path, paths in pdf_results.items():
                            image_paths.extend(paths)
                            extracted_images += len(paths)
                            
                            try:
                                os.remove(pdf_path)
                            except Exception as rm_error:
                                logger.warning(f"Could not remove PDF file {pdf_path}: {str(rm_error)}")
                        
                        pdf_total_time = time.time() - pdf_start_time
                        logger.info(f"Batch PDF processing completed in {pdf_total_time:.2f} seconds for {len(pdf_files)} PDFs")
                        task.update_progress("pdf_processing", 70, f"Extracted {extracted_images} images from {len(pdf_files)} PDF files")
                        
                    except Exception as batch_error:
                        logger.error(f"Error in batch PDF processing: {str(batch_error)}")
                        logger.info("Falling back to individual PDF processing")
                        task.update_progress("pdf_processing", 60, f"Batch processing failed, falling back to individual processing")
                        
                        for pdf_path in pdf_files:
                            try:
                                new_image_paths = process_pdf(
                                    pdf_path, 
                                    input_dir,
                                    dpi=300,  # Tăng DPI từ 100 lên 300
                                    quality=95,  # Tăng chất lượng từ 70 lên 95
                                    max_workers=12
                                )
                                image_paths.extend(new_image_paths)
                                
                                try:
                                    os.remove(pdf_path)
                                except Exception as rm_error:
                                    logger.warning(f"Could not remove PDF file {pdf_path}: {str(rm_error)}")
                                
                            except Exception as pdf_error:
                                logger.error(f"Error processing PDF file {pdf_path}: {str(pdf_error)}")
                
                # Verify we have files to process
                if not image_paths:
                    return {'error': 'No valid files were uploaded or downloaded'}, 400
                
                # Create subdirectories and distribute files
                total_files = len(image_paths)
                num_subdirs = math.ceil(total_files / batch_size)
                
                logger.info(f"Creating {num_subdirs} subdirectories for {total_files} files with batch size {batch_size}")
                
                subdir_stats = {}
                
                for i in range(num_subdirs):
                    subdir_name = f"batch_{i+1}"
                    subdir_path = os.path.join(input_dir, subdir_name)
                    os.makedirs(subdir_path, exist_ok=True)
                    
                    # Copy the template and marker files to each subdirectory
                    subdir_template_path = os.path.join(subdir_path, 'template.json')
                    subdir_marker_path = os.path.join(subdir_path, 'marker.png')
                    
                    shutil.copy2(template_path, subdir_template_path)
                    shutil.copy2(marker_path, subdir_marker_path)
                    
                    # Move files to this subdirectory
                    start_idx = i * batch_size
                    end_idx = min(start_idx + batch_size, total_files)
                    
                    subdir_files = []
                    for j in range(start_idx, end_idx):
                        file_path = image_paths[j]
                        filename = os.path.basename(file_path)
                        new_path = os.path.join(subdir_path, filename)
                        shutil.copy2(file_path, new_path)
                        subdir_files.append(filename)
                    
                    subdir_stats[subdir_name] = {
                        'file_count': end_idx - start_idx,
                        'files': subdir_files
                    }
                
                # Start processing of subdirectories
                start_time = time.time()
                
                results = []
                for subdir_name, stats in subdir_stats.items():
                    subdir_path = os.path.join(input_dir, subdir_name)
                    
                    logger.info(f"Processing subdirectory {subdir_name} with {stats['file_count']} files")
                    
                    try:
                        api_args = {
                            'input_paths': [subdir_path],
                            'output_dir': app.config['OUTPUTS_DIR_ABS'],
                            'autoAlign': False,
                            'setLayout': False,
                            'debug': True,
                        }
                        
                        tuning_config = CONFIG_DEFAULTS
                        tuning_config.dimensions.processing_width = 800
                        tuning_config.outputs.save_image_level = 0
                        tuning_config.outputs.show_image_level = 0
                        
                        subdir_template_path = os.path.join(subdir_path, 'template.json')
                        template = Template(Path(subdir_template_path), tuning_config)
                        
                        subdir_results = process_dir(
                            Path(app.config['INPUTS_DIR_ABS']),
                            Path(subdir_path),
                            api_args,
                            template=template,
                            tuning_config=tuning_config
                        )
                        
                        results.append({
                            'subdir': subdir_name,
                            'file_count': stats['file_count'],
                            'status': 'success'
                        })
                        
                    except Exception as e:
                        logger.error(f"Error processing subdirectory {subdir_name}: {str(e)}")
                        results.append({
                            'subdir': subdir_name,
                            'file_count': stats['file_count'],
                            'status': 'error',
                            'error': str(e)
                        })
                
                processing_time = time.time() - start_time
                
                response_data = {
                    'message': 'Batch processing completed',
                    'directory_name': directory_name,
                    'total_files': total_files,
                    'batch_size': batch_size,
                    'subdirectories': len(subdir_stats),
                    'template_path': template_path,
                    'marker_path': marker_path,
                    'template_from_url': bool(template_url),
                    'marker_from_url': bool(marker_url),
                    'processing_time': round(processing_time, 2),
                    'results': results
                }
                
                # Clean up if requested
                if clean_after:
                    try:
                        if os.path.exists(input_dir):
                            shutil.rmtree(input_dir)
                        
                        if os.path.exists(output_dir):
                            shutil.rmtree(output_dir)
                    except Exception as e:
                        logger.warning(f"Error cleaning directories after processing: {str(e)}")
                
                return response_data, 200
                
            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
                return {'error': f'Error processing batch: {str(e)}'}, 500

    return api 