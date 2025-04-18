"""
Request parsers for API endpoints
"""
from flask_restx import reqparse, inputs
from werkzeug.datastructures import FileStorage

def setup_parsers():
    """Initialize request parsers for API endpoints"""
    
    upload_parser = reqparse.RequestParser()
    upload_parser.add_argument('template_file', 
                              location='files',
                              type=FileStorage, 
                              required=False,
                              help='JSON template file')
    upload_parser.add_argument('marker_file', 
                              location='files',
                              type=FileStorage, 
                              required=False,
                              help='Marker image file for template (PNG, JPG, JPEG)')
    upload_parser.add_argument('template_url', 
                              type=str, 
                              required=False,
                              location='form',
                              help='URL to template JSON file')
    upload_parser.add_argument('marker_url', 
                              type=str, 
                              required=False,
                              location='form',
                              help='URL to marker image file (PNG, JPG, JPEG)')
    upload_parser.add_argument('image_files', 
                              location='files',
                              type=FileStorage, 
                              required=False, 
                              action='append',
                              help='OMR image files (PNG, JPG, JPEG, PDF)')
    upload_parser.add_argument('file_urls', 
                              type=str, 
                              required=False,
                              location='form',
                              action='append',
                              help='URLs to OMR image files (PNG, JPG, JPEG, PDF)')
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
    upload_parser.add_argument('batch_size', 
                              type=int, 
                              required=False,
                              default=50,
                              help='Number of files per batch for batch processing')

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
    
    # Files upload parser for generic file uploads
    files_upload_parser = reqparse.RequestParser()
    files_upload_parser.add_argument('files',
                              location='files',
                              type=FileStorage,
                              required=True,
                              action='append',
                              help='Files to upload')
    
    # Single file upload parser - simpler version
    single_file_parser = reqparse.RequestParser()
    single_file_parser.add_argument('file',
                              location='files',
                              type=FileStorage,
                              required=True,
                              help='Single file to upload')
    
    return {
        'upload_parser': upload_parser,
        'get_params_parser': get_params_parser,
        'files_upload_parser': files_upload_parser,
        'single_file_parser': single_file_parser
    } 