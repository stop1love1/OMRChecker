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
from pathlib import Path
import pandas as pd

from flask import send_file, send_from_directory, Blueprint, url_for, request
from flask_restx import Api, Resource
from flask_swagger_ui import get_swaggerui_blueprint
from werkzeug.utils import secure_filename

from src.entry import process_dir
from src.defaults import CONFIG_DEFAULTS
from src.template import Template
from src.logger import logger

from utils.validators import validate_directory_name, clean_nan_values, force_string_conversion
from utils.file_handling import transform_result_format, save_to_public_images, clean_all_folders, download_file_from_url, clean_old_files
from utils.image_processing import process_pdf, validate_image, safe_resize
from api.models import setup_models
from api.parsers import setup_parsers

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
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response
    
    # Ensure all required directories exist
    os.makedirs(app.config['INPUTS_DIR_ABS'], exist_ok=True)
    os.makedirs(app.config['OUTPUTS_DIR_ABS'], exist_ok=True)
    os.makedirs(app.config['PROCESSED_DIR'], exist_ok=True)
    os.makedirs(app.config['PUBLIC_IMAGES_DIR_ABS'], exist_ok=True)
    logger.info(f"Ensured all required directories exist")
    
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
    
    # Define API resources
    @ns.route('/process-omr')
    @ns.expect(parsers['upload_parser'])
    class ProcessOMR(Resource):
        @ns.doc('process_omr', 
                responses={
                    200: 'Success',
                    400: 'Validation Error',
                    500: 'Processing Error'
                })
        def post(self):
            """Process multiple OMR sheets with the provided template"""
            try:
                args = parsers['upload_parser'].parse_args()
                
                directory_name = args['directory_name']
                
                clean_before = args['clean_before']
                if isinstance(clean_before, str):
                    clean_before = clean_before.lower() != 'false'
                
                clean_after = args['clean_after']
                if isinstance(clean_after, str):
                    clean_after = clean_after.lower() != 'false'
                
                logger.info(f"Processing OMR for directory: {directory_name}")
                
                is_valid, error_message = validate_directory_name(directory_name)
                if not is_valid:
                    return {"error": error_message}, 400
                
                image_files = args['image_files'] or []
                file_urls = args['file_urls'] or []
                
                if not image_files and not file_urls:
                    return {'error': 'At least one image file or file URL must be provided'}, 400
                
                # Ensure INPUTS_DIR_ABS exists
                os.makedirs(app.config['INPUTS_DIR_ABS'], exist_ok=True)
                
                input_dir = os.path.join(app.config['INPUTS_DIR_ABS'], directory_name)
                
                if clean_before and os.path.exists(input_dir):
                    try:
                        shutil.rmtree(input_dir)
                    except Exception as e:
                        logger.warning(f"Error cleaning input directory: {str(e)}")
                
                # Create input directory if it doesn't exist
                os.makedirs(input_dir, exist_ok=True)
                
                # Ensure OUTPUTS_DIR_ABS exists
                os.makedirs(app.config['OUTPUTS_DIR_ABS'], exist_ok=True)
                
                output_dir = os.path.join(app.config['OUTPUTS_DIR_ABS'], directory_name)
                
                if clean_before and os.path.exists(output_dir):
                    try:
                        shutil.rmtree(output_dir)
                    except Exception as e:
                        logger.warning(f"Error cleaning output directory: {str(e)}")
                
                # Create output directory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)
                
                try:
                    template_file = args['template_file']
                    if not template_file.filename.endswith('.json'):
                        return {'error': 'Template file must be a JSON file'}, 400
                    
                    template_path = os.path.join(input_dir, 'template.json')
                    template_file.save(template_path)
                    
                    marker_file = args['marker_file']
                    if marker_file and not any(marker_file.filename.lower().endswith(ext) 
                                  for ext in ['.png', '.jpg', '.jpeg']):
                        return {'error': f'Marker file {marker_file.filename} must be a PNG, JPG, or JPEG file'}, 400
                    
                    if marker_file:
                        marker_path = os.path.join(input_dir, 'marker.png')
                        marker_file.save(marker_path)
                        
                        if not os.path.exists(marker_path):
                            logger.error(f"Failed to save marker file at {marker_path}")
                    
                    pdf_files = []
                    regular_image_files = []
                    
                    for image_file in image_files:
                        if image_file.filename.lower().endswith('.pdf'):
                            logger.info(f"Found PDF file: {image_file.filename}")
                            pdf_path = os.path.join(input_dir, image_file.filename)
                            image_file.save(pdf_path)
                            pdf_files.append(pdf_path)
                        else:
                            image_path = os.path.join(input_dir, image_file.filename)
                            image_file.save(image_path)
                            regular_image_files.append(image_path)
                    
                    # Process URL files
                    for file_url in file_urls:
                        # Log full URL for debugging
                        logger.info(f"Processing URL: {file_url}")
                        
                        # Check if URL ends with space or has encoding issues
                        if file_url.strip() != file_url:
                            logger.warning(f"URL contains leading/trailing spaces, cleaning: '{file_url}'")
                            file_url = file_url.strip()

                        downloaded_path = download_file_from_url(file_url, input_dir, app.config)
                        
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
                    for img_path in image_paths:
                        is_valid, error_message = validate_image(img_path)
                        
                        if is_valid:
                            valid_image_paths.append(img_path)
                        else:
                            logger.warning(f"Skipping invalid image: {error_message}")
                    
                    # Update image_paths with only valid images
                    image_paths = valid_image_paths
                    
                    if pdf_files:
                        logger.info(f"Processing {len(pdf_files)} PDF files in batch mode")
                        pdf_start_time = time.time()
                        
                        try:
                            from utils.image_processing import process_pdf_batch
                            from utils.batch_config import get_batch_profile
                            
                            batch_profile = get_batch_profile(len(pdf_files))
                            
                            pdf_results = process_pdf_batch(
                                pdf_files, 
                                input_dir,
                                dpi=batch_profile["dpi"],
                                quality=batch_profile["quality"]
                            )
                            
                            for pdf_path, paths in pdf_results.items():
                                image_paths.extend(paths)
                                
                                try:
                                    os.remove(pdf_path)
                                except Exception as rm_error:
                                    logger.warning(f"Could not remove PDF file {pdf_path}: {str(rm_error)}")
                            
                            pdf_total_time = time.time() - pdf_start_time
                            logger.info(f"Batch PDF processing completed in {pdf_total_time:.2f} seconds for {len(pdf_files)} PDFs")
                            
                        except Exception as batch_error:
                            logger.error(f"Error in batch PDF processing: {str(batch_error)}")
                            logger.info("Falling back to individual PDF processing")
                            
                            from utils.image_processing import process_pdf
                            
                            for pdf_path in pdf_files:
                                try:
                                    new_image_paths = process_pdf(
                                        pdf_path, 
                                        input_dir,
                                        dpi=100,
                                        quality=70,
                                        max_workers=12
                                    )
                                    image_paths.extend(new_image_paths)
                                    
                                    try:
                                        os.remove(pdf_path)
                                    except Exception as rm_error:
                                        logger.warning(f"Could not remove PDF file {pdf_path}: {str(rm_error)}")
                                    
                                except Exception as pdf_error:
                                    logger.error(f"Error processing PDF file {pdf_path}: {str(pdf_error)}")
                    
                    if not image_paths:
                        return {'error': 'No valid images to process'}, 400
                                    
                    api_args = {
                        'input_paths': [input_dir],
                        'output_dir': app.config['OUTPUTS_DIR_ABS'],
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
                        tuning_config.dimensions.processing_width = 800
                        tuning_config.outputs.save_image_level = 0
                        tuning_config.outputs.show_image_level = 0
                    
                    template = Template(Path(template_path), tuning_config)
                    
                    root_dir = Path(app.config['INPUTS_DIR_ABS'])
                    curr_dir = Path(input_dir)
                    
                    omr_start_time = time.time()
                    
                    results = process_dir(
                        root_dir,
                        curr_dir,
                        api_args,
                        template=template,
                        tuning_config=tuning_config
                    )
                    
                    output_dir_path = Path(output_dir)
                    
                    all_csv_files = list(output_dir_path.glob('**/*.csv'))
                    
                    results_file = list(output_dir_path.glob('**/CheckedOMRs/*.csv'))
                    
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
                    
                    if not results_file:
                        return {'error': 'Processing failed - no results generated'}, 500
                    
                    csv_file_path = results_file[0]
                    
                    df = pd.read_csv(csv_file_path, dtype={'studentId': str, 'code': str})
                    
                    df = force_string_conversion(df, ['studentId', 'code'])
                    
                    df = df.replace([np.inf, -np.inf], 'Infinity')
                    df = df.fillna("")
                    
                    results_data = df.to_dict(orient='records')
                    
                    result_id = str(uuid.uuid4())
                    result_dir = Path(app.config['PROCESSED_DIR']) / result_id
                    os.makedirs(result_dir, exist_ok=True)
                    
                    for file in output_dir_path.glob('**/*'):
                        if file.is_file():
                            try:
                                rel_path = file.relative_to(output_dir_path)
                                target_path = result_dir / rel_path
                                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                                shutil.copy2(file, target_path)
                            except Exception as copy_error:
                                logger.warning(f"Error copying file {file}: {str(copy_error)}")
                    
                    clean_results = []
                    for result in results_data:
                        clean_result = clean_nan_values(result)
                        transformed_result = transform_result_format(clean_result)
                        
                        # Find input and output image paths
                        input_image_path = None
                        output_image_path = None
                        
                        if 'input_path' in clean_result:
                            input_image_path = clean_result['input_path']
                            # Verify the path exists and is valid
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
                            # Verify the path exists and is valid
                            if output_image_path and not os.path.exists(output_image_path):
                                logger.warning(f"Output image path does not exist: {output_image_path}")
                                output_image_path = None
                        
                        # Save images to public directory and add URLs to result
                        public_input_image = None
                        public_output_image = None
                        
                        # Ensure public images directory exists
                        os.makedirs(app.config['PUBLIC_IMAGES_DIR_ABS'], exist_ok=True)
                        
                        if input_image_path and os.path.exists(input_image_path):
                            public_input_image = save_to_public_images(
                                input_image_path, 
                                "input", 
                                app.config['API_HOST'], 
                                app.config['PUBLIC_IMAGES_DIR_ABS']
                            )
                            if public_input_image:
                                transformed_result['input_image_url'] = public_input_image
                        
                        if output_image_path and os.path.exists(output_image_path):
                            public_output_image = save_to_public_images(
                                output_image_path, 
                                "output", 
                                app.config['API_HOST'], 
                                app.config['PUBLIC_IMAGES_DIR_ABS']
                            )
                            if public_output_image:
                                transformed_result['output_image_url'] = public_output_image
                        
                        clean_results.append(transformed_result)
                    
                    end_time = time.time()
                    total_processing_time = end_time - omr_start_time
                    logger.info(f"Total processing completed in {total_processing_time:.2f} seconds")
                    
                    response_data = {
                        'message': 'OMR processing completed successfully',
                        'result_id': result_id,
                        'input_dir': str(input_dir),
                        'output_dir': str(output_dir),
                        'csv_file': str(csv_file_path.name),
                        'timing': {
                            'total_processing': round(total_processing_time, 2)
                        },
                        'results': clean_results
                    }
                    
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
                    logger.error(f"Error processing OMR: {str(e)}")
                    return {'error': f'Error processing OMR: {str(e)}'}, 500

            except Exception as e:
                logger.error(f"Error processing OMR: {str(e)}")
                return {'error': f'Error processing OMR: {str(e)}'}, 500

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
            """Process large batches of files by splitting into subfolders of 50 files each"""
            try:
                args = parsers['upload_parser'].parse_args()
                
                directory_name = args['directory_name']
                
                clean_before = args['clean_before']
                if isinstance(clean_before, str):
                    clean_before = clean_before.lower() != 'false'
                
                clean_after = args['clean_after']
                if isinstance(clean_after, str):
                    clean_after = clean_after.lower() != 'false'
                
                batch_size = args.get('batch_size', 50)
                if isinstance(batch_size, str):
                    try:
                        batch_size = int(batch_size)
                    except ValueError:
                        batch_size = 50
                
                # Validate batch size
                if batch_size < 1 or batch_size > 200:
                    return {"error": "Batch size must be between 1 and 200"}, 400
                    
                logger.info(f"Processing batch OMR for directory: {directory_name} with batch size {batch_size}")
                
                is_valid, error_message = validate_directory_name(directory_name)
                if not is_valid:
                    return {"error": error_message}, 400
                
                image_files = args['image_files'] or []
                file_urls = args['file_urls'] or []
                
                if not image_files and not file_urls:
                    return {'error': 'At least one image file or file URL must be provided'}, 400
                
                # Ensure base directories exist
                base_input_dir = os.path.join(app.config['INPUTS_DIR_ABS'], directory_name)
                base_output_dir = os.path.join(app.config['OUTPUTS_DIR_ABS'], directory_name)
                
                if clean_before:
                    # Clean directories if they exist
                    if os.path.exists(base_input_dir):
                        try:
                            shutil.rmtree(base_input_dir)
                        except Exception as e:
                            logger.warning(f"Error cleaning input directory: {str(e)}")
                    
                    if os.path.exists(base_output_dir):
                        try:
                            shutil.rmtree(base_output_dir)
                        except Exception as e:
                            logger.warning(f"Error cleaning output directory: {str(e)}")
                
                # Create base directories
                os.makedirs(base_input_dir, exist_ok=True)
                os.makedirs(base_output_dir, exist_ok=True)
                
                # Create initial holding directory for all uploaded files
                uploads_dir = os.path.join(base_input_dir, "_all_uploads")
                os.makedirs(uploads_dir, exist_ok=True)
                
                # Process uploaded files and URLs
                all_files = []
                
                # Handle template and marker files
                try:
                    template_file = args['template_file']
                    if not template_file.filename.endswith('.json'):
                        return {'error': 'Template file must be a JSON file'}, 400
                    
                    template_path = os.path.join(uploads_dir, 'template.json')
                    template_file.save(template_path)
                    
                    marker_file = args['marker_file']
                    if marker_file and not any(marker_file.filename.lower().endswith(ext) 
                                  for ext in ['.png', '.jpg', '.jpeg']):
                        return {'error': f'Marker file {marker_file.filename} must be a PNG, JPG, or JPEG file'}, 400
                    
                    if marker_file:
                        marker_path = os.path.join(uploads_dir, 'marker.png')
                        marker_file.save(marker_path)
                        
                        if not os.path.exists(marker_path):
                            logger.error(f"Failed to save marker file at {marker_path}")
                
                except Exception as e:
                    logger.error(f"Error handling template/marker files: {str(e)}")
                    return {'error': f'Error handling template files: {str(e)}'}, 500
                
                # Save uploaded image files
                for image_file in image_files:
                    image_path = os.path.join(uploads_dir, image_file.filename)
                    image_file.save(image_path)
                    all_files.append(image_path)
                
                # Process URL files
                for file_url in file_urls:
                    # Clean URL if needed
                    if file_url.strip() != file_url:
                        file_url = file_url.strip()
                    
                    downloaded_path = download_file_from_url(file_url, uploads_dir, app.config)
                    
                    if downloaded_path:
                        all_files.append(downloaded_path)
                    else:
                        logger.warning(f"Failed to download file from URL: {file_url}")
                
                # Validate all files before processing
                valid_files = []
                for file_path in all_files:
                    # PDF files don't need image validation
                    if file_path.lower().endswith('.pdf'):
                        valid_files.append(file_path)
                        continue
                        
                    is_valid, error_message = validate_image(file_path)
                    if is_valid:
                        valid_files.append(file_path)
                    else:
                        logger.warning(f"Skipping invalid file: {os.path.basename(file_path)} - {error_message}")
                
                if not valid_files:
                    return {'error': 'No valid files to process'}, 400
                
                # Organize files into subfolders
                from utils.image_processing import organize_files_into_subfolders, process_subfolder_batches, speed_optimized_batch_processor
                
                batch_start_time = time.time()
                
                # Xử lý nhanh với song song
                processing_results = speed_optimized_batch_processor(
                    valid_files, 
                    base_input_dir, 
                    base_output_dir,
                    files_per_folder=batch_size,
                    max_workers=min(os.cpu_count(), 4)  # Giới hạn 4 luồng
                )
                
                # Build response data
                merged_dir = processing_results.get('merged_dir', '')
                merged_csv_path = os.path.join(merged_dir, "merged_results.csv")
                
                # Read merged results if available
                results_data = []
                if os.path.exists(merged_csv_path):
                    try:
                        df = pd.read_csv(merged_csv_path, dtype={'studentId': str, 'code': str})
                        df = force_string_conversion(df, ['studentId', 'code'])
                        df = df.replace([np.inf, -np.inf], 'Infinity')
                        df = df.fillna("")
                        results_data = df.to_dict(orient='records')
                    except Exception as e:
                        logger.error(f"Error reading merged results: {str(e)}")
                
                # Clean up results data
                clean_results = []
                for result in results_data:
                    clean_result = clean_nan_values(result)
                    transformed_result = transform_result_format(clean_result)
                    clean_results.append(transformed_result)
                
                batch_end_time = time.time()
                batch_total_time = batch_end_time - batch_start_time
                
                # Create response data
                response_data = {
                    'message': 'Batch processing completed successfully',
                    'directory_name': directory_name,
                    'input_dir': str(base_input_dir),
                    'output_dir': str(base_output_dir),
                    'processed_files': {
                        'total_files': len(valid_files),
                        'total_subfolders': processing_results.get('total_subfolders', 0),
                        'total_csv_files': processing_results.get('total_csv_files', 0),
                        'total_pdf_pages': processing_results.get('total_pdf_pages', 0)
                    },
                    'timing': {
                        'total_processing': round(batch_total_time, 2)
                    },
                    'results': clean_results
                }
                
                # Clean up if requested
                if clean_after:
                    try:
                        # First clean uploads directory
                        if os.path.exists(uploads_dir):
                            shutil.rmtree(uploads_dir)
                            
                        # Don't clean output directory as it contains results
                    except Exception as e:
                        logger.warning(f"Error cleaning directories after processing: {str(e)}")
                
                return response_data, 200
                
            except Exception as e:
                logger.error(f"Error in batch processing: {str(e)}")
                return {'error': f'Error in batch processing: {str(e)}'}, 500

    return api 