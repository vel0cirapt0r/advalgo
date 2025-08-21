import logging
import os
from datetime import datetime
import colorama
from colorama import Fore, Style
from constants import DEBUG_MODE

class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to log messages based on their level."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.color_map = {
            logging.DEBUG: Fore.BLUE,
            logging.INFO: Fore.GREEN,
            logging.WARNING: Fore.YELLOW,
            logging.ERROR: Fore.RED,
            logging.CRITICAL: Fore.RED + Style.BRIGHT,
        }

    def format(self, record):
        s = super().format(record)
        color = self.color_map.get(record.levelno, '')
        return color + s + Style.RESET_ALL


def setup_logging(log_file=f'log/{datetime.now()}.log', level=logging.DEBUG):
    """Configure the root logger with file and console handlers."""
    # Initialize Colorama for cross-platform color support
    colorama.init()

    # Create log directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Set up file handler with standard formatter
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s.%(funcName)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)  # Always log DEBUG and above to file

    # Set up console handler with colored formatter
    console_handler = logging.StreamHandler()
    console_formatter = ColoredFormatter(
        '%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s.%(funcName)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)

    # Set console handler level based on DEBUG_MODE in constants.py
    try:
        if DEBUG_MODE:
            console_handler.setLevel(logging.DEBUG)
        else:
            console_handler.setLevel(logging.INFO)
    except AttributeError:
        # Default to INFO if DEBUG_MODE is not defined in constants.py
        console_handler.setLevel(logging.INFO)

    # Configure the root logger
    root_logger = logging.getLogger()
    if not root_logger.handlers:  # Avoid duplicate handlers
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        root_logger.setLevel(level)
