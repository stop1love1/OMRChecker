#!/usr/bin/env python3
"""
Run the OMRChecker API server
"""

import sys
from api_server import app

def print_ascii_banner():
    banner = """
   ____  __  __ ____    ____ _               _             
  / __ \|  \/  |  _ \  / ___| |__   ___  ___| | _____ _ __ 
 | |  | | |\/| | |_) || |   | '_ \ / _ \/ __| |/ / _ \ '__|
 | |__| | |  | |  _ < | |___| | | |  __/ (__|   <  __/ |   
  \____/|_|  |_|_| \_(_)____|_| |_|\___|\___|_|\_\___|_|   
                                                            
  API Server v1.0
"""
    print(banner)

if __name__ == "__main__":
    # Print banner and info
    print_ascii_banner()
    
    # Get port from command line or use default
    port = 5000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port number: {sys.argv[1]}. Using default port 5000.")
    
    # Print available endpoints
    host = '0.0.0.0'
    base_url = f"http://localhost:{port}"
    
    print("\nAvailable endpoints:")
    print(f"  • Web Interface:     {base_url}/")
    print(f"  • Enhanced Swagger:  {base_url}/swagger")
    print(f"  • API Documentation: {base_url}/api/docs")
    print(f"  • Health Check:      {base_url}/api/health")
    
    print("\nStarting OMRChecker API server...")
    print(f"Server running at {base_url}")
    print("Press Ctrl+C to stop the server.")
    
    # Start the server
    app.run(debug=True, host=host, port=port) 