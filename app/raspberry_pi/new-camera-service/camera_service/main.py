# camera_service/main.py
import os
from .config.settings import Settings
from .api.routes import create_app
from .config.logging_config import configure_logging

def main():
    configure_logging()
    
    # Load Solana environment file if it exists
    solana_env_path = os.path.join(os.path.dirname(__file__), ".env.solana")
    if os.path.exists(solana_env_path):
        Settings.load_env_file(solana_env_path)
    
    Settings.setup()
    app = create_app()
    app.run(host=Settings.CAMERA_API_HOST, port=Settings.CAMERA_API_PORT, debug=False)

if __name__ == "__main__":
    main()