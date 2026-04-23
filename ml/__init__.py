"""
RecursiaDx ML Module

Medical image analysis module for:
- Malaria detection (blood smear)
- Platelet counting (blood smear)
- Brain tumor detection (MRI via Brain Tumor API)
"""

__version__ = "1.0.0"
__author__ = "RecursiaDx Team"

# Import main classes for easy access
from .models.malaria_predictor import MalariaPredictor
from .models.platelet_counter import PlateletCounter
from .utils.data_manager import DataManager
from .config.config import get_config

__all__ = [
    'MalariaPredictor',
    'PlateletCounter',
    'DataManager', 
    'get_config'
]