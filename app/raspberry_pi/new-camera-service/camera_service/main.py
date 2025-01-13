# camera_service/main.py
from .config.settings import Settings
from .api.routes import create_app
from .config.logging_config import configure_logging

def main():
    configure_logging()
    Settings.setup()
    app = create_app()
    app.run(host=Settings.CAMERA_API_HOST, port=Settings.CAMERA_API_PORT, debug=False)

if __name__ == "__main__":
    main()