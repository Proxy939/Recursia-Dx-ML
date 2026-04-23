"""
RecursiaDx ML Module

Medical image analysis module for:
- Brain tumor detection (MRI via Brain Tumor API on port 5002)
- Pneumonia detection (Chest X-ray via Pneumonia API on port 5003)
"""

__version__ = "1.1.0"
__author__ = "RecursiaDx Team"

# Import main classes for easy access
from .utils.data_manager import DataManager
from .config.config import get_config

__all__ = [
    'DataManager', 
    'get_config'
]