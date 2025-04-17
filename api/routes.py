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

from flask import send_file, send_from_directory, Blueprint
from flask_restx import Api, Resource
from flask_swagger_ui import get_swaggerui_blueprint

from src.entry import process_dir
from src.defaults import CONFIG_DEFAULTS
from src.template import Template
from src.logger import logger

from utils.validators import validate_directory_name, clean_nan_values, force_string_conversion
from utils.file_handling import transform_result_format, save_to_public_images, clean_all_folders
from utils.image_processing import process_pdf
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
    
    # Define routes
    @app.route('/')
    def index():
        return send_from_directory('static', 'index.html')
    
    @app.route('/static/<path:path>')
    def send_static(path):
        return send_from_directory('static', path)
    
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
                
                # Convert string values to boolean if needed
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
                
                input_dir = os.path.join(app.config['INPUTS_DIR_ABS'], directory_name)
                
                if clean_before and os.path.exists(input_dir):
                    try:
                        shutil.rmtree(input_dir)
                    except Exception as e:
                        logger.warning(f"Error cleaning input directory: {str(e)}")
                
                os.makedirs(input_dir, exist_ok=True)
                
                output_dir = os.path.join(app.config['OUTPUTS_DIR_ABS'], directory_name)
                
                if clean_before and os.path.exists(output_dir):
                    try:
                        shutil.rmtree(output_dir)
                    except Exception as e:
                        logger.warning(f"Error cleaning output directory: {str(e)}")
                
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
                        # Save marker file with fixed name regardless of original filename
                        marker_path = os.path.join(input_dir, 'marker.png')
                        marker_file.save(marker_path)
                        
                        if not os.path.exists(marker_path):
                            logger.error(f"Failed to save marker file at {marker_path}")
                    
                    image_paths = []
                    
                    for image_file in args['image_files']:
                        if image_file.filename.lower().endswith('.pdf'):
                            logger.info(f"Processing PDF file: {image_file.filename}")
                            
                            pdf_path = os.path.join(input_dir, image_file.filename)
                            image_file.save(pdf_path)
                            
                            try:
                                # Optimize PDF processing with much lower DPI and more parallelism
                                new_image_paths = process_pdf(
                                    pdf_path, 
                                    input_dir,
                                    dpi=100,  # Lower DPI for speed
                                    quality=70,  # Lower quality for faster processing
                                    max_workers=12  # More parallel processing
                                )
                                image_paths.extend(new_image_paths)
                                
                                # Remove the original PDF to save space
                                try:
                                    os.remove(pdf_path)
                                except Exception as rm_error:
                                    logger.warning(f"Could not remove PDF file {pdf_path}: {str(rm_error)}")
                                
                            except Exception as pdf_error:
                                logger.error(f"Error processing PDF file {image_file.filename}: {str(pdf_error)}")
                                return {'error': f'Error processing PDF file {image_file.filename}: {str(pdf_error)}. Check if poppler-utils is correctly installed.'}, 500
                        
                        else:
                            image_path = os.path.join(input_dir, image_file.filename)
                            image_file.save(image_path)
                            image_paths.append(image_path)
                    
                    # Setup arguments for OMR processing with optimized performance settings
                    api_args = {
                        'input_paths': [input_dir],
                        'output_dir': app.config['OUTPUTS_DIR_ABS'],
                        'autoAlign': False,
                        'setLayout': False,
                        'debug': True,
                    }
                    
                    # Use optimized tuning configuration for faster processing
                    tuning_config = CONFIG_DEFAULTS
                    
                    # Set optimized processing parameters
                    tuning_config.dimensions.processing_width = 800  # Lower resolution processing
                    tuning_config.outputs.save_image_level = 0  # Reduce image saving for speed
                    tuning_config.outputs.show_image_level = 0  # Disable image display for speed
                    
                    template = Template(Path(template_path), tuning_config)
                    
                    # Use Path objects consistently for process_dir
                    root_dir = Path(app.config['INPUTS_DIR_ABS'])
                    curr_dir = Path(input_dir)
                    
                    # Measure OMR processing time
                    omr_start_time = time.time()
                    
                    results = process_dir(
                        root_dir,
                        curr_dir,
                        api_args,
                        template=template,
                        tuning_config=tuning_config
                    )
                    
                    output_dir_path = Path(output_dir)
                    
                    # Find CSV result file with fallback options
                    all_csv_files = list(output_dir_path.glob('**/*.csv'))
                    
                    # Look in CheckedOMRs directory first
                    results_file = list(output_dir_path.glob('**/CheckedOMRs/*.csv'))
                    
                    # Then look in Results folder
                    if not results_file:
                        results_file = list(output_dir_path.glob('**/Results/*.csv'))
                    
                    # Try Results_*.csv anywhere
                    if not results_file:
                        results_file = list(output_dir_path.glob('**/Results_*.csv'))
                    
                    # Last resort: any CSV except ErrorFiles.csv
                    if not results_file:
                        non_error_csv = [f for f in all_csv_files if 'ErrorFiles.csv' not in str(f)]
                        if non_error_csv:
                            results_file = [non_error_csv[0]]
                        else:
                            results_file = all_csv_files
                    
                    if not results_file:
                        return {'error': 'Processing failed - no results generated'}, 500
                    
                    csv_file_path = results_file[0]
                    
                    # Read CSV with studentId and code always as strings
                    df = pd.read_csv(csv_file_path, dtype={'studentId': str, 'code': str})
                    
                    # Apply additional string conversion to ensure leading zeros are preserved
                    df = force_string_conversion(df, ['studentId', 'code'])
                    
                    # Replace NaN and infinite values
                    df = df.replace([np.inf, -np.inf], 'Infinity')
                    df = df.fillna("")
                    
                    results_data = df.to_dict(orient='records')
                    
                    # Create unique ID for this result set
                    result_id = str(uuid.uuid4())
                    result_dir = Path(app.config['PROCESSED_DIR']) / result_id
                    os.makedirs(result_dir, exist_ok=True)
                    
                    # Copy result files to persistent directory
                    for file in output_dir_path.glob('**/*'):
                        if file.is_file():
                            try:
                                rel_path = file.relative_to(output_dir_path)
                                target_path = result_dir / rel_path
                                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                                shutil.copy2(file, target_path)
                            except Exception as copy_error:
                                logger.warning(f"Error copying file {file}: {str(copy_error)}")
                    
                    # Process results and save public images
                    clean_results = []
                    for result in results_data:
                        clean_result = clean_nan_values(result)
                        transformed_result = transform_result_format(clean_result)
                        
                        # Find input and output image paths
                        input_image_path = None
                        output_image_path = None
                        
                        if 'input_path' in clean_result:
                            input_image_path = clean_result['input_path']
                        elif 'file_id' in clean_result:
                            file_id = clean_result['file_id']
                            for img_path in image_paths:
                                if os.path.basename(img_path) == file_id:
                                    input_image_path = img_path
                                    break
                        
                        if 'output_path' in clean_result:
                            output_image_path = clean_result['output_path']
                        
                        # Save images to public directory and add URLs to result
                        public_input_image = None
                        public_output_image = None
                        
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
                    
                    # Calculate total processing time
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
                    
                    # Clean directories if requested
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
            
            # Convert string values to boolean if needed
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
            
            # Find CSV result file with multiple fallback options
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
            
            # Read CSV with studentId and code always as strings
            df = pd.read_csv(csv_files[0], dtype={'studentId': str, 'code': str})
            
            # Apply additional string conversion to ensure leading zeros are preserved
            df = force_string_conversion(df, ['studentId', 'code'])
            
            # Replace NaN and infinite values
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
            # Security check for path traversal
            if '..' in filename or filename.startswith('/'):
                return {'error': 'Invalid filename'}, 400
                
            result_dir = Path(app.config['PROCESSED_DIR']) / result_id
            
            if not result_dir.exists():
                return {'error': 'Result not found'}, 404
            
            # Special handling for Results_11AM.csv
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
            
            # Additional security check for path traversal
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

    # Return API instance for reference
    return api 