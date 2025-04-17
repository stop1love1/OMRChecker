"""
API endpoints for file upload and management
"""
import os
import uuid
import shutil
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = Path('uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf', 'tif', 'tiff'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max upload size

# Create uploads folder if it doesn't exist
UPLOAD_FOLDER.mkdir(exist_ok=True)

# Configure the Flask app
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_unique_filename(filename):
    """Generate a unique filename with UUID and timestamp"""
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    secure_name = secure_filename(filename)
    name, ext = os.path.splitext(secure_name)
    return f"{name}_{now}_{unique_id}{ext}"

@app.route('/api/upload', methods=['POST'])
def upload_files():
    """
    Upload multiple files endpoint
    
    Returns:
        JSON response with upload results
    """
    if 'files' not in request.files:
        return jsonify({
            'success': False,
            'message': 'No files provided',
            'files': []
        }), 400
    
    files = request.files.getlist('files')
    if len(files) == 0:
        return jsonify({
            'success': False,
            'message': 'No files selected',
            'files': []
        }), 400
    
    uploaded_files = []
    errors = []
    
    for file in files:
        if file and allowed_file(file.filename):
            original_filename = secure_filename(file.filename)
            unique_filename = generate_unique_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            try:
                file.save(file_path)
                file_url = request.host_url + 'uploads/' + unique_filename
                
                uploaded_files.append({
                    'originalName': original_filename,
                    'fileName': unique_filename,
                    'url': file_url,
                    'size': os.path.getsize(file_path),
                    'path': file_path
                })
            except Exception as e:
                errors.append({
                    'filename': original_filename,
                    'error': str(e)
                })
        else:
            errors.append({
                'filename': file.filename if file.filename else 'unknown',
                'error': 'Invalid file type or empty file'
            })
    
    return jsonify({
        'success': len(uploaded_files) > 0,
        'message': f'Successfully uploaded {len(uploaded_files)} files. Failed: {len(errors)}',
        'files': uploaded_files,
        'errors': errors
    })

@app.route('/api/delete/<filename>', methods=['DELETE'])
def delete_file(filename):
    """
    Delete a specific file
    
    Args:
        filename: Filename to delete
        
    Returns:
        JSON response with delete result
    """
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(file_path):
        return jsonify({
            'success': False,
            'message': f'File not found: {filename}'
        }), 404
    
    try:
        os.remove(file_path)
        return jsonify({
            'success': True,
            'message': f'File deleted: {filename}'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error deleting file: {str(e)}'
        }), 500

@app.route('/api/delete-all', methods=['DELETE'])
def delete_all_files():
    """
    Delete all files in the uploads directory
    
    Returns:
        JSON response with delete result
    """
    try:
        # Count files before deletion
        files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) 
                if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], f))]
        file_count = len(files)
        
        # Delete all files in the directory
        for file in files:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
            os.remove(file_path)
            
        return jsonify({
            'success': True,
            'message': f'All files deleted ({file_count} files)',
            'count': file_count
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error deleting files: {str(e)}'
        }), 500

@app.route('/uploads/<filename>')
def serve_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/files', methods=['GET'])
def list_files():
    """
    List all files in the uploads directory
    
    Returns:
        JSON response with file list
    """
    files = []
    try:
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(file_path):
                file_url = request.host_url + 'uploads/' + filename
                file_stats = os.stat(file_path)
                files.append({
                    'fileName': filename,
                    'url': file_url,
                    'size': file_stats.st_size,
                    'modifiedDate': datetime.fromtimestamp(file_stats.st_mtime).isoformat()
                })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error listing files: {str(e)}',
            'files': []
        }), 500
    
    return jsonify({
        'success': True,
        'message': f'Found {len(files)} files',
        'files': files
    })

@app.route('/api/process', methods=['POST'])
def process_files():
    """
    Process uploaded files with OMRChecker
    
    Expected JSON payload:
    {
        "files": ["filename1.jpg", "filename2.jpg"],
        "options": {
            "setLayout": false,
            "outputDir": "results"
        }
    }
    
    Returns:
        JSON response with processing results
    """
    if not request.json or 'files' not in request.json:
        return jsonify({
            'success': False,
            'message': 'No files specified in the request'
        }), 400
    
    filenames = request.json['files']
    options = request.json.get('options', {})
    
    input_dir = app.config['UPLOAD_FOLDER']
    output_dir = options.get('outputDir', 'results')
    
    # Validate files exist
    valid_files = []
    for filename in filenames:
        file_path = os.path.join(input_dir, filename)
        if os.path.exists(file_path):
            valid_files.append(file_path)
    
    if not valid_files:
        return jsonify({
            'success': False,
            'message': 'None of the specified files exist'
        }), 400
    
    # Prepare arguments for OMRChecker
    args = {
        'output_dir': output_dir,
        'setLayout': options.get('setLayout', False)
    }
    
    try:
        # Import OMRChecker entry point
        from simple_entry import entry_point
        
        # Run OMRChecker on the files
        result = entry_point(input_dir, args)
        
        return jsonify({
            'success': True,
            'message': 'Processing completed',
            'result': result,
            'processedFiles': len(valid_files)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error processing files: {str(e)}'
        }), 500

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Enable CORS for development
    from flask_cors import CORS
    CORS(app)
    
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True) 