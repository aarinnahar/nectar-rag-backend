import logging  
import os

def setup_logging():
    logger = logging.getLogger("app")  
    logger.setLevel(logging.DEBUG)  

    # Get the directory where THIS file (logging.py) exists
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Create full path to app.log inside that directory
    log_file = os.path.join(base_dir, "app.log")


    # Console handler  
    console_handler = logging.StreamHandler()  
    console_handler.setLevel(logging.INFO)  

    # File handler  
    file_handler = logging.FileHandler(log_file, mode= "w")  
    file_handler.setLevel(logging.DEBUG)  

    # Formatter  
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )  

    console_handler.setFormatter(formatter)  
    file_handler.setFormatter(formatter)  

    logger.addHandler(console_handler)  
    logger.addHandler(file_handler)  

    logger.info("App started")  
    logger.debug("Debug details")
