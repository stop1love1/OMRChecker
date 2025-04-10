"""
OMRChecker API Server

This module provides a REST API with Swagger integration for the OMRChecker.
"""

import os
import json
import tempfile
import shutil
import uuid
import base64
import math
import numpy as np
import re
from pathlib import Path
from werkzeug.datastructures import FileStorage

from flask import Flask, request, send_from_directory, send_file, render_template, Blueprint, jsonify
from flask_restx import Api, Resource, fields, reqparse
from flask_swagger_ui import get_swaggerui_blueprint
import cv2

from src.entry import process_dir
from src.defaults import CONFIG_DEFAULTS
from src.template import Template
from src.logger import logger

# Setup Flask app
app = Flask(__name__)
app.static_folder = 'static'  # Explicitly set static folder
app.config['PROCESSED_DIR'] = os.path.join(tempfile.gettempdir(), 'omrchecker_results')
app.config['INPUTS_DIR'] = 'inputs'  # Set inputs directory
app.config['OUTPUTS_DIR'] = 'outputs'  # Set outputs directory
os.makedirs(app.config['PROCESSED_DIR'], exist_ok=True)
os.makedirs(app.config['INPUTS_DIR'], exist_ok=True)  # Ensure inputs directory exists
os.makedirs(app.config['OUTPUTS_DIR'], exist_ok=True)  # Ensure outputs directory exists

# Convert paths to absolute paths to avoid path issues
app.config['INPUTS_DIR_ABS'] = os.path.abspath(app.config['INPUTS_DIR'])
app.config['OUTPUTS_DIR_ABS'] = os.path.abspath(app.config['OUTPUTS_DIR'])

# Setup API with Swagger UI
authorizations = {
    'apikey': {
        'type': 'apiKey',
        'in': 'header',
        'name': 'Authorization'
    }
}

# Create a blueprint for API
blueprint = Blueprint('api', __name__, url_prefix='/api')
api = Api(
    blueprint,
    version='1.0',
    title='OMRChecker API',
    description='API for OMR sheet processing',
    doc='/docs',
    authorizations=authorizations
)

# Register the blueprint
app.register_blueprint(blueprint)

# Create enhanced Swagger UI
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

# Create namespaces
ns = api.namespace('', description='OMR operations')

# Model definitions
template_model = api.model('Template', {
    'description': fields.String(description='Template description'),
    'author': fields.String(description='Template author'),
})

omr_result_model = api.model('OMRResult', {
    'file_name': fields.String(description='Image file name'),
    'result_id': fields.String(description='Unique result identifier'),
    'message': fields.String(description='Processing status message'),
})

# Upload parser for template file
upload_parser = reqparse.RequestParser()
upload_parser.add_argument('template_file', 
                          location='files',
                          type=FileStorage, 
                          required=True,
                          help='JSON template file')
upload_parser.add_argument('marker_file', 
                          location='files',
                          type=FileStorage, 
                          required=False,
                          help='Marker image file for template (PNG, JPG, JPEG)')
upload_parser.add_argument('image_files', 
                          location='files',
                          type=FileStorage, 
                          required=True, 
                          action='append',
                          help='OMR image files (PNG, JPG, JPEG)')
upload_parser.add_argument('directory_name', 
                          type=str, 
                          required=True,
                          help='Name of the directory to create (no "/" allowed)')
upload_parser.add_argument('include_images', 
                          type=bool, 
                          required=False,
                          default=False,
                          help='Include base64 encoded processed images in response')
upload_parser.add_argument('clean_before', 
                          type=bool, 
                          required=False,
                          default=True,
                          help='Clean directories before processing')
upload_parser.add_argument('clean_after', 
                          type=bool, 
                          required=False,
                          default=False,
                          help='Clean directories after processing and saving results')

# Helper function to validate directory name
def validate_directory_name(directory_name):
    """
    Validates that the directory name does not contain path separators or invalid characters
    Returns (is_valid, error_message)
    """
    if '/' in directory_name or '\\' in directory_name:
        return False, "Directory name cannot contain path separators ('/' or '\\')"
    
    # Additional validation for other invalid characters
    if not re.match(r'^[a-zA-Z0-9_\-\.]+$', directory_name):
        return False, "Directory name can only contain alphanumeric characters, underscores, hyphens, and periods"
    
    return True, None

# Helper function to handle NaN values in JSON
def clean_nan_values(obj):
    """Convert NaN, Infinity, -Infinity to null or strings in JSON objects"""
    if isinstance(obj, dict):
        return {k: clean_nan_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_values(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj):
            return ""  # Convert NaN to empty string
        if math.isinf(obj):
            return str(obj)  # Convert Infinity/-Infinity to strings
    return obj

# Custom JSON encoder for handling NaN values
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float):
            if math.isnan(obj):
                return ""
            if math.isinf(obj):
                return str(obj)
        return super().default(obj)

# Register custom JSON encoder
app.json_encoder = CustomJSONEncoder

# Serve the HTML frontend
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

# Helper function to transform result data
def transform_result_format(result_data):
    """
    Transform flat result object with q1, q2... keys into a structured format
    with metadata and an answers array containing key-value pairs
    """
    if not result_data or not isinstance(result_data, dict):
        return result_data
    
    # Extract metadata fields (non-question fields)
    metadata = {k: v for k, v in result_data.items() 
               if not k.startswith('q') or not k[1:].replace('-', '').isdigit()}
    
    # Extract question fields and create answers array
    answers = []
    for key, value in result_data.items():
        if key.startswith('q') and key[1:].replace('-', '').isdigit():
            answers.append({
                "key": key,
                "value": value
            })
    
    # Create transformed result
    transformed_result = metadata.copy()
    transformed_result['answers'] = answers
    
    return transformed_result

@ns.route('/process-omr')
@ns.expect(upload_parser)
class ProcessOMR(Resource):
    @ns.doc('process_omr', 
            responses={
                200: 'Success',
                400: 'Validation Error',
                500: 'Processing Error'
            })
    def post(self):
        """
        Process multiple OMR sheets with the provided template
        
        This endpoint accepts a template JSON file and multiple OMR images for processing.
        It returns the processed results for each image and a unique ID for retrieving all results later.
        """
        args = upload_parser.parse_args()
        
        template_file = args['template_file']
        marker_file = args['marker_file']
        image_files = args['image_files']
        directory_name = args['directory_name']
        include_images = args['include_images']
        clean_before = args['clean_before']
        clean_after = args['clean_after']
        
        # Validate template file
        if not template_file.filename.endswith('.json'):
            return {'error': 'Template file must be a JSON file'}, 400
        
        # Validate marker file if provided
        if marker_file and not any(marker_file.filename.lower().endswith(ext) 
                      for ext in ['.png', '.jpg', '.jpeg']):
            return {'error': f'Marker file {marker_file.filename} must be a PNG, JPG, or JPEG file'}, 400
        
        # Validate image files
        for image_file in image_files:
            if not any(image_file.filename.lower().endswith(ext) 
                      for ext in ['.png', '.jpg', '.jpeg']):
                return {'error': f'Image file {image_file.filename} must be a PNG, JPG, or JPEG file'}, 400
        
        # Validate directory name
        is_valid, error_message = validate_directory_name(directory_name)
        if not is_valid:
            return {'error': error_message}, 400
        
        # Create input directory structure - use absolute paths
        input_dir = os.path.join(app.config['INPUTS_DIR_ABS'], directory_name)
        
        # Clean input directory if requested
        if clean_before and os.path.exists(input_dir):
            try:
                logger.info(f"Cleaning input directory: {input_dir}")
                shutil.rmtree(input_dir)
            except Exception as e:
                logger.warning(f"Error cleaning input directory: {str(e)}")
        
        os.makedirs(input_dir, exist_ok=True)
        
        # Set output directory path - use absolute paths
        output_dir = os.path.join(app.config['OUTPUTS_DIR_ABS'], directory_name)
        
        # Clean output directory if requested
        if clean_before and os.path.exists(output_dir):
            try:
                logger.info(f"Cleaning output directory: {output_dir}")
                shutil.rmtree(output_dir)
            except Exception as e:
                logger.warning(f"Error cleaning output directory: {str(e)}")
        
        try:
            # Save template file - always name it template.json
            template_path = os.path.join(input_dir, 'template.json')
            template_file.save(template_path)
            logger.info(f"Saved template file to {template_path}")
            
            # Save marker file if provided
            if marker_file:
                marker_path = os.path.join(input_dir, marker_file.filename)
                marker_file.save(marker_path)
                logger.info(f"Saved marker file to {marker_path}")
            
            # Save all image files with original filenames
            image_paths = []
            for image_file in image_files:
                image_path = os.path.join(input_dir, image_file.filename)
                image_file.save(image_path)
                image_paths.append(image_path)
                logger.info(f"Saved image file to {image_path}")
            
            # Process the OMR sheets
            api_args = {
                'input_paths': [input_dir],
                'output_dir': app.config['OUTPUTS_DIR_ABS'],  # Set parent output dir, process_dir will create subdirectory
                'autoAlign': False,
                'setLayout': False,
                'debug': True,
            }
            
            # Load template
            tuning_config = CONFIG_DEFAULTS
            template = Template(Path(template_path), tuning_config)
            
            # Process the directory
            logger.info(f"Processing directory: {input_dir}")
            
            # For process_dir, use Path objects consistently
            root_dir = Path(app.config['INPUTS_DIR_ABS'])
            curr_dir = Path(input_dir)
            
            results = process_dir(
                root_dir,
                curr_dir,
                api_args,
                template=template,
                tuning_config=tuning_config
            )
            
            # Look for Results_11AM.csv file
            output_dir_path = Path(output_dir)
            
            # Log all available CSV files for debugging
            all_csv_files = list(output_dir_path.glob('**/*.csv'))
            logger.info(f"All CSV files found in output directory: {[str(f) for f in all_csv_files]}")
            
            # First check for CSV files in CheckedOMRs directory (where actual results are)
            results_file = list(output_dir_path.glob('**/CheckedOMRs/*.csv'))
            if results_file:
                logger.info(f"Found CSV in CheckedOMRs: {results_file[0]}")
            
            # Then check for Results_XXX.csv in the Results folder
            if not results_file:
                results_file = list(output_dir_path.glob('**/Results/*.csv'))
                if results_file:
                    logger.info(f"Found CSV in Results folder: {results_file[0]}")
            
            # If Results/*.csv not found, look for Results_*.csv anywhere
            if not results_file:
                results_file = list(output_dir_path.glob('**/Results_*.csv'))
                if results_file:
                    logger.info(f"Found Results_*.csv: {results_file[0]}")
            
            # Last resort: any CSV file (excluding ErrorFiles.csv if possible)
            if not results_file:
                non_error_csv = [f for f in all_csv_files if 'ErrorFiles.csv' not in str(f)]
                if non_error_csv:
                    results_file = [non_error_csv[0]]
                    logger.info(f"Using non-error CSV as fallback: {results_file[0]}")
                else:
                    results_file = all_csv_files
                    if results_file:
                        logger.info(f"Using any CSV file as last resort: {results_file[0]}")
            
            if not results_file:
                return {'error': 'Processing failed - no results generated'}, 500
            
            # Read the results file
            import pandas as pd
            csv_file_path = results_file[0]
            logger.info(f"Reading CSV results from: {csv_file_path}")
            
            # Read CSV and explicitly handle NaN values
            df = pd.read_csv(csv_file_path)
            logger.info(f"CSV contents: {df.head().to_dict()}")
            logger.info(f"CSV shape: {df.shape}")
            
            # Replace NaN and infinite values with empty strings
            df = df.replace([np.inf, -np.inf], 'Infinity')  # Replace infinity with string
            df = df.fillna("")  # Replace NaN with empty string
            
            # Get all results
            results_data = df.to_dict(orient='records')
            
            # Create a unique ID for this result set
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
            
            # Clean NaN values and transform all results
            clean_results = []
            for result in results_data:
                clean_result = clean_nan_values(result)
                transformed_result = transform_result_format(clean_result)
                clean_results.append(transformed_result)
            
            response_data = {
                'message': 'OMR processing completed successfully',
                'result_id': result_id,
                'input_dir': str(input_dir),
                'output_dir': str(output_dir),
                'csv_file': str(csv_file_path.name),
                'results': clean_results  # Array of results
            }
            
            # Include processed images if requested
            if include_images:
                images = {}
                for img_file in result_dir.glob('**/*.png'):
                    try:
                        rel_path = img_file.relative_to(result_dir)
                        with open(img_file, 'rb') as f:
                            img_data = base64.b64encode(f.read()).decode('utf-8')
                            images[str(rel_path)] = img_data
                    except Exception as img_error:
                        logger.warning(f"Error processing image {img_file}: {str(img_error)}")
                
                response_data['images'] = images
            
            # Clean directories after processing if requested
            if clean_after:
                try:
                    if os.path.exists(input_dir):
                        logger.info(f"Cleaning input directory after processing: {input_dir}")
                        shutil.rmtree(input_dir)
                    
                    if os.path.exists(output_dir):
                        logger.info(f"Cleaning output directory after processing: {output_dir}")
                        shutil.rmtree(output_dir)
                except Exception as e:
                    logger.warning(f"Error cleaning directories after processing: {str(e)}")
            
            # Return JSON response with proper encoding
            return response_data, 200
            
        except Exception as e:
            logger.error(f"Error processing OMR: {str(e)}")
            return {'error': f'Error processing OMR: {str(e)}'}, 500

@ns.route('/results/<string:result_id>')
@ns.param('result_id', 'The unique identifier for the result set')
class Results(Resource):
    @ns.doc('get_results', 
            responses={
                200: 'Success', 
                404: 'Result not found'
            })
    def get(self, result_id):
        """
        Get results for a specific processed OMR
        
        Retrieves the stored results using a previously generated result ID.
        """
        result_dir = Path(app.config['PROCESSED_DIR']) / result_id
        
        if not result_dir.exists():
            return {'error': 'Result not found'}, 404
        
        # First check for CSV files in CheckedOMRs directory
        csv_files = list(result_dir.glob('**/CheckedOMRs/*.csv'))
        
        if not csv_files:
            # Then check for Results_XXX.csv in the Results folder
            csv_files = list(result_dir.glob('**/Results/*.csv')) 
        
        if not csv_files:
            # If Results/*.csv not found, look for Results_*.csv anywhere
            csv_files = list(result_dir.glob('**/Results_*.csv'))
        
        if not csv_files:
            # If not found, look for any CSV file (excluding ErrorFiles.csv if possible)
            all_csv = list(result_dir.glob('**/*.csv'))
            non_error_csv = [f for f in all_csv if 'ErrorFiles.csv' not in str(f)]
            if non_error_csv:
                csv_files = [non_error_csv[0]]
            else:
                csv_files = all_csv
                
        if not csv_files:
            return {'error': 'No results found'}, 404
        
        import pandas as pd
        # Read CSV and explicitly handle NaN values
        df = pd.read_csv(csv_files[0])
        # Replace NaN and infinite values with empty strings
        df = df.replace([np.inf, -np.inf], 'Infinity')  # Replace infinity with string
        df = df.fillna("")  # Replace NaN with empty string
        
        # Get all results
        results_data = df.to_dict(orient='records')
        
        # Clean NaN values and transform all results
        clean_results = []
        for result in results_data:
            clean_result = clean_nan_values(result)
            transformed_result = transform_result_format(clean_result)
            clean_results.append(transformed_result)
        
        return {
            'result_id': result_id,
            'csv_file': csv_files[0].name,
            'results': clean_results  # Return array of results
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
        """
        Download a file from the results
        
        Downloads a specific file (CSV, image, etc.) from a previously processed result set.
        """
        # Validate filename doesn't contain path traversal
        if '..' in filename or filename.startswith('/'):
            return {'error': 'Invalid filename'}, 400
            
        result_dir = Path(app.config['PROCESSED_DIR']) / result_id
        
        if not result_dir.exists():
            return {'error': 'Result not found'}, 404
        
        # Handle special case for 'Results_11AM.csv'
        if filename.lower() == 'results_11am.csv' or filename.lower() == 'results.csv':
            # Try to find Results_11AM.csv
            results_file = list(result_dir.glob('**/Results_11AM.csv'))
            if results_file:
                file_path = results_file[0]
            else:
                # Fall back to any CSV file
                results_file = list(result_dir.glob('**/*.csv'))
                if not results_file:
                    return {'error': 'No CSV results found'}, 404
                file_path = results_file[0]
        else:
            # Regular file path
            file_path = result_dir / filename
            
        if not file_path.exists() or not file_path.is_file():
            return {'error': 'File not found'}, 404
        
        # Validate the file is within the result_dir (no path traversal)
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
        """
        Health check endpoint
        
        Simple endpoint to check if the API is running correctly.
        """
        return {'status': 'healthy'}, 200

# Generate Swagger JSON
@app.route('/api/swagger.json')
def swagger_json():
    schema_json = api.__schema__
    # Need to manually serialize to ensure NaN handling
    return app.response_class(
        json.dumps(schema_json, cls=CustomJSONEncoder),
        mimetype='application/json'
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 