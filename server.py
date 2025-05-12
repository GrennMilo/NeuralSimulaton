import http.server
import socketserver
import os
import webbrowser
from urllib.parse import urlparse, parse_qs

# Port to serve on
PORT = 8000

# Create custom request handler
class NeuralNetworkHandler(http.server.SimpleHTTPRequestHandler):
    # Set content types based on file extensions
    def guess_type(self, path):
        base, ext = os.path.splitext(path)
        if ext == '.js':
            return 'application/javascript'
        elif ext == '.css':
            return 'text/css'
        elif ext == '.html':
            return 'text/html'
        else:
            return super().guess_type(path)
    
    # Log requests
    def log_message(self, format, *args):
        print(f"[SERVER] {' '.join(str(arg) for arg in args)}")
        
# Create and start server
def run_server():
    print(f"Starting Neural Network Simulator on http://localhost:{PORT}")
    print("Press Ctrl+C to stop the server")
    
    # Change to the directory where this script is located
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        # Create the server
        with socketserver.TCPServer(("", PORT), NeuralNetworkHandler) as httpd:
            # Open browser
            webbrowser.open(f"http://localhost:{PORT}")
            # Start serving
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_server()