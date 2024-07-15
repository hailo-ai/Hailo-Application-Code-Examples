import logging
import os
import sys


class ConsoleFormatter(logging.Formatter):
    """
    Logging Formatter for the console.
    """

    COLOR_PREFIX = '\x1b['
    COLOR_SUFFIX = '\x1b[0m'
    COLORS = {
        logging.DEBUG: '36m',  # blue
        logging.INFO: '32m',  # green
        logging.WARNING: '33;1m',  # bold yellow
        logging.ERROR: '31;1m',  # bold red
        logging.CRITICAL: '41;1m',  # bold white on red
    }

    def format(self, record):
        level_name = record.levelname
        level_no = record.levelno
        message = '%(message)s'
        if sys.stdout.isatty():
            level_fmt = f'[{self.COLOR_PREFIX}{self.COLORS[level_no]}{level_name.lower()}{self.COLOR_SUFFIX}]'
        else:
            level_fmt = f'[{level_name.lower()}]'
        formatter = logging.Formatter(f'{level_fmt} {message}')
        return formatter.format(record)


def init_logger(logger_name, log_path, log_level=logging.INFO, debug_log=True):
    logger = logging.Logger(logger_name)
    
    console_fh = logging.StreamHandler(sys.stdout)
    console_fh.setFormatter(ConsoleFormatter())
    console_fh.set_name('stream_handler')
    console_fh.setLevel(log_level)
    logger.addHandler(console_fh)
    
    fmt = '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s'
    file_formatter = logging.Formatter(fmt)
    
    fhandler = logging.FileHandler(log_path)
    fhandler.setLevel(log_level)
    fhandler.setFormatter(file_formatter)
    logger.addHandler(fhandler)
    
    if debug_log:
        basepath, basename = os.path.split(log_path)
        debug_name = ".debug."+basename
        debug_path = os.path.join(basepath, debug_name)
        dhandler = logging.FileHandler(debug_path)
        dhandler.setLevel(logging.DEBUG)
        dhandler.setFormatter(file_formatter)
        logger.addHandler(dhandler)
    
    return logger
