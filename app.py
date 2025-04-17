"""
OMRChecker API Server - Main application file
"""

import os
import tempfile
from flask import Flask

from api import setup_routes
from scheduler import setup_scheduler
from src.logger import logger

def create_app():
    """Create and configure the Flask application"""
    
    app = Flask(__name__)
    app.static_folder = 'static'
    
    # Configure directories
    app.config['PROCESSED_DIR'] = os.path.join(tempfile.gettempdir(), 'omrchecker_results')
    app.config['INPUTS_DIR'] = 'inputs'
    app.config['OUTPUTS_DIR'] = 'outputs'
    app.config['PUBLIC_IMAGES_DIR'] = 'images'
    app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024  # 1000 MB max upload

    # Create directories if they don't exist
    os.makedirs(app.config['PROCESSED_DIR'], exist_ok=True)
    os.makedirs(app.config['INPUTS_DIR'], exist_ok=True)
    os.makedirs(app.config['OUTPUTS_DIR'], exist_ok=True)
    os.makedirs(app.config['PUBLIC_IMAGES_DIR'], exist_ok=True) 

    # Set absolute paths for directories
    app.config['INPUTS_DIR_ABS'] = os.path.abspath(app.config['INPUTS_DIR'])
    app.config['OUTPUTS_DIR_ABS'] = os.path.abspath(app.config['OUTPUTS_DIR'])
    app.config['PUBLIC_IMAGES_DIR_ABS'] = os.path.abspath(app.config['PUBLIC_IMAGES_DIR'])

    # Configure API host from environment or default
    app.config['API_HOST'] = os.environ.get('API_HOST', 'http://localhost:5000')
    
    # Set up routes and API
    api = setup_routes(app)
    
    # Set up background scheduler
    scheduler = setup_scheduler(app.config)
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000) 