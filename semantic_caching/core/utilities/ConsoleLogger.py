import logging
import time

logger = logging.getLogger(__name__)

class ConsoleLogger(object):

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.message = None

    def start_timer(self, message: str = ""):
        self.message = f"[INFO] {message}"
        self.start_time = time.time()
        print(self.message, end='', flush=True)
        logger.info(self.message)

    def end_timer(self, message: str = ""):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        duration = round(duration, 2)

        if not message: 
            message = f"complete ({str(duration)}s"

        new_message = f'\r{self.message} {message})'
        print(new_message)
        logger.info(new_message)

    def add_message(self, message): 
        self.message = f"{self.message} {message}"
