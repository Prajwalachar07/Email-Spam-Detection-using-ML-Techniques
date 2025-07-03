from app import app  # Import your existing app from app.py

def handler(environ, start_response):
    return app.wsgi_app(environ, start_response)