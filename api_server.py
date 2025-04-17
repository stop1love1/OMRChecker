"""
OMRChecker API Server - Legacy file

This file is kept for backward compatibility.
The code has been refactored into a more maintainable structure.
Please use app.py/run.py for the new implementation.
"""

from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 