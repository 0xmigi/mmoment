from camera_service.main import create_app
from camera_service.config.settings import Settings

app = create_app()

if __name__ == "__main__":
    Settings.setup()
    app.run(host="127.0.0.1", port=8000) 