"""
Global warning suppression module.
Import this at the very beginning of any script to suppress PyPDF2 font warnings.
"""
import warnings
import logging
import sys

# Suppress ALL warnings from PyPDF2
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*FontBBox.*')
warnings.filterwarnings('ignore', message='.*font descriptor.*')

# Set PyPDF2 logging to CRITICAL only
logging.getLogger('PyPDF2').setLevel(logging.CRITICAL)
logging.getLogger('pdfminer').setLevel(logging.CRITICAL)
logging.getLogger('PIL').setLevel(logging.CRITICAL)

# Redirect stderr to suppress any remaining warnings
class SuppressWarnings:
    def __init__(self):
        self._original_stderr = sys.stderr
        
    def write(self, text):
        # Filter out font warnings
        if 'FontBBox' not in text and 'font descriptor' not in text:
            self._original_stderr.write(text)
    
    def flush(self):
        self._original_stderr.flush()

# Only suppress in production, not during development
# Uncomment the next line to completely suppress stderr warnings:
# sys.stderr = SuppressWarnings()
