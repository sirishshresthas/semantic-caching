import logging
import os


def setup_logging(log_file='cache.log', level=logging.INFO, log_format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
    """
    Sets up the logging configuration for the package.
    
    Parameters:
    - log_file: The name of the log file to write to.
    - level: The logging level (e.g., logging.INFO, logging.DEBUG).
    - log_format: The format for the log messages.
    """
    log_directory = './logs'
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    log_file_path = os.path.join(log_directory, log_file)
    
    logging.basicConfig(filename=log_file_path, 
                        level=level, 
                        format=log_format,
                        filemode='a') 

    
    logging.info("Logging configured.")
