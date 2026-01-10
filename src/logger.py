import logging
import os
from datetime import datetime

# Absolute path to project root
BASE_DIR = os.getcwd()

# Logs directory
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Log file name
LOG_FILE = f"log_{datetime.now().strftime('%Y-%m-%d')}.log"

# Full path
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    logging.info("Logging has started successfully")
