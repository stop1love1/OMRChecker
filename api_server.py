"""
OMRChecker API Server
"""

import os
import json
import tempfile
import shutil
import uuid
import math
import numpy as np
import re
import time
import datetime
from pathlib import Path
from werkzeug.datastructures import FileStorage
from apscheduler.schedulers.background import BackgroundScheduler

from flask import Flask, send_from_directory, send_file, Blueprint
from flask_restx import Api, Resource, fields, reqparse, inputs
from flask_swagger_ui import get_swaggerui_blueprint
import fitz

from src.entry import process_dir
from src.defaults import CONFIG_DEFAULTS
from src.template import Template
from src.logger import logger

app = Flask(__name__)
app.static_folder = 'static'
app.config['PROCESSED_DIR'] = os.path.join(tempfile.gettempdir(), 'omrchecker_results')
app.config['INPUTS_DIR'] = 'inputs'
app.config['OUTPUTS_DIR'] = 'outputs'
app.config['PUBLIC_IMAGES_DIR'] = 'images'
app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024

os.makedirs(app.config['PROCESSED_DIR'], exist_ok=True)
os.makedirs(app.config['INPUTS_DIR'], exist_ok=True)
os.makedirs(app.config['OUTPUTS_DIR'], exist_ok=True)
os.makedirs(app.config['PUBLIC_IMAGES_DIR'], exist_ok=True) 

app.config['INPUTS_DIR_ABS'] = os.path.abspath(app.config['INPUTS_DIR'])
app.config['OUTPUTS_DIR_ABS'] = os.path.abspath(app.config['OUTPUTS_DIR'])
app.config['PUBLIC_IMAGES_DIR_ABS'] = os.path.abspath(app.config['PUBLIC_IMAGES_DIR'])

app.config['API_HOST'] = os.environ.get('API_HOST', 'http://localhost:5000')

authorizations = {
    'apikey': {
        'type': 'apiKey',
        'in': 'header',
        'name': 'Authorization'
    }
}

blueprint = Blueprint('api', __name__, url_prefix='/api')
api = Api(
    blueprint,
    version='1.0',
    title='OMRChecker API',
    description='API for OMR sheet processing',
    doc='/docs',
    authorizations=authorizations
)

app.register_blueprint(blueprint)

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

ns = api.namespace('', description='OMR operations')

template_model = api.model('Template', {
    'description': fields.String(description='Template description'),
    'author': fields.String(description='Template author'),
})

omr_result_model = api.model('OMRResult', {
    'file_name': fields.String(description='Image file name'),
    'result_id': fields.String(description='Unique result identifier'),
    'message': fields.String(description='Processing status message'),
})

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
                          help='OMR image files (PNG, JPG, JPEG, PDF)')
upload_parser.add_argument('directory_name', 
                          type=str, 
                          required=True,
                          help='Name of the directory to create (no "/" allowed)')
upload_parser.add_argument('clean_before', 
                          type=inputs.boolean, 
                          required=False,
                          default=True,
                          help='Clean directories before processing')
upload_parser.add_argument('clean_after', 
                          type=inputs.boolean, 
                          required=False,
                          default=False,
                          help='Clean directories after processing and saving results')

get_params_parser = reqparse.RequestParser()
get_params_parser.add_argument('directory_name', 
                          type=str, 
                          required=True,
                          help='Name of the directory to create (no "/" allowed)')
get_params_parser.add_argument('clean_before', 
                          type=inputs.boolean, 
                          required=False,
                          default=True,
                          help='Clean directories before processing')
get_params_parser.add_argument('clean_after', 
                          type=inputs.boolean, 
                          required=False,
                          default=False,
                          help='Clean directories after processing and saving results')

def validate_directory_name(directory_name):
    """Validates that directory name doesn't contain path separators or invalid characters"""
    if '/' in directory_name or '\\' in directory_name:
        return False, "Directory name cannot contain path separators ('/' or '\\')"
    
    if not re.match(r'^[a-zA-Z0-9_\-\.]+$', directory_name):
        return False, "Directory name can only contain alphanumeric characters, underscores, hyphens, and periods"
    
    return True, None

def clean_nan_values(obj):
    """Replace NaN, Infinity with empty strings or string representations"""
    if isinstance(obj, dict):
        # Ensure studentId and code are always strings
        if 'studentId' in obj and obj['studentId'] is not None:
            obj['studentId'] = str(obj['studentId'])
        if 'code' in obj and obj['code'] is not None:
            obj['code'] = str(obj['code'])
        return {k: clean_nan_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_values(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj):
            return "" 
        if math.isinf(obj):
            return str(obj)
    return obj

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float):
            if math.isnan(obj):
                return ""
            if math.isinf(obj):
                return str(obj)
        return super().default(obj)

app.json_encoder = CustomJSONEncoder

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/images/<filename>')
def serve_public_image(filename):
    return send_from_directory(app.config['PUBLIC_IMAGES_DIR_ABS'], filename)

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

def save_to_public_images(image_path, prefix):
    """Save image to public directory with unique filename and return its URL"""
    try:
        timestamp = int(time.time())
        unique_id = str(uuid.uuid4())[:8]
        file_ext = os.path.splitext(image_path)[1].lower()
        
        filename = f"{prefix}_{timestamp}_{unique_id}{file_ext}"
        public_path = os.path.join(app.config['PUBLIC_IMAGES_DIR_ABS'], filename)
        
        shutil.copy2(image_path, public_path)
        
        return f"{app.config['API_HOST']}/images/{filename}"
    except Exception as e:
        logger.error(f"Failed to save public image: {e}")
        return None

public_image_model = api.model('PublicImage', {
    'url': fields.String(description='URL to access image from outside')
})

result_model = api.model('Result', {
    'file_id': fields.String(description='Original image filename'),
    'score': fields.Float(description='Score'),
    'studentId': fields.String(description='Student ID'),
    'code': fields.String(description='Code or identifier'),
    'input_image_url': fields.String(description='URL of input image'),
    'output_image_url': fields.String(description='URL of processed image'),
    'answers': fields.List(fields.Nested(api.model('Answer', {
        'key': fields.String(description='Question code'),
        'value': fields.String(description='Answer value')
    }))),
})

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
        """Process multiple OMR sheets with the provided template"""
        try:
            args = upload_parser.parse_args()
            
            directory_name = args['directory_name']
            
            # Convert string values to boolean if needed
            clean_before = args['clean_before']
            if isinstance(clean_before, str):
                clean_before = clean_before.lower() != 'false'
            
            clean_after = args['clean_after']
            if isinstance(clean_after, str):
                clean_after = clean_after.lower() != 'false'
            
            logger.info(f"Processing OMR for directory: {directory_name}")
            
            if not validate_directory_name(directory_name):
                return {"error": "Invalid directory name. Use alphanumeric characters, underscore, and hyphen only."}, 400
            
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
                            pdf_processed = False
                            # Try PyMuPDF first
                            try:
                                pdf_document = fitz.open(pdf_path)
                                
                                if pdf_document.page_count == 0:
                                    logger.warning(f"PDF file has no pages: {image_file.filename}")
                                    
                                for i in range(pdf_document.page_count):
                                    page = pdf_document[i]
                                    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                                    img_filename = f"{os.path.splitext(image_file.filename)[0]}_page_{i+1}.jpg"
                                    img_path = os.path.join(input_dir, img_filename)
                                    pix.save(img_path)
                                    image_paths.append(img_path)
                                    
                                pdf_document.close()
                                pdf_processed = True
                                
                            except ImportError as e:
                                logger.warning(f"PyMuPDF not available: {str(e)}. Will try pdf2image...")
                            except Exception as e:
                                logger.warning(f"Error using PyMuPDF: {str(e)}. Will try pdf2image...")
                                
                            # Fall back to pdf2image if PyMuPDF failed
                            if not pdf_processed:
                                try:
                                    from pdf2image import convert_from_path
                                    
                                    try:
                                        pdf_images = convert_from_path(pdf_path)
                                    except Exception as poppler_err:
                                        logger.warning(f"Default poppler path failed: {str(poppler_err)}. Trying alternate paths...")
                                        
                                        # Try different poppler paths for different systems
                                        poppler_paths = [
                                            '/usr/bin',
                                            '/usr/local/bin',
                                            '/opt/homebrew/bin',  # MacOS Homebrew
                                            '/usr/lib/x86_64-linux-gnu'
                                        ]
                                        
                                        for poppler_path in poppler_paths:
                                            try:
                                                pdf_images = convert_from_path(pdf_path, poppler_path=poppler_path)
                                                break
                                            except Exception as path_err:
                                                logger.warning(f"Failed with poppler path {poppler_path}: {str(path_err)}")
                                        else:
                                            raise Exception("All poppler paths failed. Make sure poppler-utils is correctly installed.")
                                    
                                    for i, image in enumerate(pdf_images):
                                        img_filename = f"{os.path.splitext(image_file.filename)[0]}_page_{i+1}.jpg"
                                        img_path = os.path.join(input_dir, img_filename)
                                        image.save(img_path, 'JPEG')
                                        image_paths.append(img_path)
                                    
                                    pdf_processed = True
                                    
                                except ImportError as e:
                                    logger.error(f"pdf2image not available: {str(e)}")
                                    raise ImportError("Neither PyMuPDF nor pdf2image are installed. Please install at least one of these libraries to process PDF files: 'pip install PyMuPDF pdf2image'")
                                except Exception as e:
                                    logger.error(f"Error processing PDF with pdf2image: {str(e)}")
                                    raise
                            
                            if not pdf_processed:
                                raise Exception("Failed to process PDF with any available method")
                                
                        except Exception as pdf_error:
                            logger.error(f"Error processing PDF file {image_file.filename}: {str(pdf_error)}")
                            return {'error': f'Error processing PDF file {image_file.filename}: {str(pdf_error)}. Check if poppler-utils is correctly installed.'}, 500
                        
                        try:
                            os.remove(pdf_path)
                        except Exception as rm_error:
                            logger.warning(f"Could not remove PDF file {pdf_path}: {str(rm_error)}")
                    
                    else:
                        image_path = os.path.join(input_dir, image_file.filename)
                        image_file.save(image_path)
                        image_paths.append(image_path)
                
                # Setup arguments for OMR processing
                api_args = {
                    'input_paths': [input_dir],
                    'output_dir': app.config['OUTPUTS_DIR_ABS'],
                    'autoAlign': False,
                    'setLayout': False,
                    'debug': True,
                }
                
                tuning_config = CONFIG_DEFAULTS
                template = Template(Path(template_path), tuning_config)
                
                # Use Path objects consistently for process_dir
                root_dir = Path(app.config['INPUTS_DIR_ABS'])
                curr_dir = Path(input_dir)
                
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
                
                import pandas as pd
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
                        public_input_image = save_to_public_images(input_image_path, "input")
                        if public_input_image:
                            transformed_result['input_image_url'] = public_input_image
                    
                    if output_image_path and os.path.exists(output_image_path):
                        public_output_image = save_to_public_images(output_image_path, "output")
                        if public_output_image:
                            transformed_result['output_image_url'] = public_output_image
                    
                    clean_results.append(transformed_result)
                
                response_data = {
                    'message': 'OMR processing completed successfully',
                    'result_id': result_id,
                    'input_dir': str(input_dir),
                    'output_dir': str(output_dir),
                    'csv_file': str(csv_file_path.name),
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
        args = get_params_parser.parse_args()
        
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
        
        import pandas as pd
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
            total_items = clean_all_folders()
            return {
                'status': 'success',
                'message': f'Deleted {total_items} items from all directories'
            }, 200
        except Exception as e:
            logger.error(f"Error during manual cleaning: {str(e)}")
            return {'error': f'Error cleaning directories: {str(e)}'}, 500

@app.route('/api/swagger.json')
def swagger_json():
    schema_json = api.__schema__
    return app.response_class(
        json.dumps(schema_json, cls=CustomJSONEncoder),
        mimetype='application/json'
    )

# Define a function to ensure leading zeros are preserved in string fields
def force_string_conversion(data, string_fields):
    """
    Ensure all specified fields are converted to strings with leading zeros preserved.
    This handles cases where pandas might have already converted strings to numbers.
    """
    for field in string_fields:
        if field in data.columns:
            # Convert to string and ensure leading zeros are preserved
            data[field] = data[field].fillna('')
            data[field] = data[field].apply(lambda x: str(x) if x != '' else '')
    return data

# Function to clean directory contents
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

def clean_all_folders():
    """Delete contents of all directories: inputs, outputs, images"""
    total_items = 0
    
    total_items += clean_folder_contents(app.config['INPUTS_DIR_ABS'], "inputs")
    
    total_items += clean_folder_contents(app.config['OUTPUTS_DIR_ABS'], "outputs")
    
    total_items += clean_folder_contents(app.config['PUBLIC_IMAGES_DIR_ABS'], "images")
    
    logger.info(f"Cleaned a total of {total_items} items from all directories")
    return total_items

# Setup scheduler for cleaning all folders
scheduler = BackgroundScheduler()
scheduler.add_job(clean_all_folders, 'cron', hour=0, minute=0)
scheduler.start()

# Stop scheduler when application exits
import atexit
atexit.register(lambda: scheduler.shutdown())

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 