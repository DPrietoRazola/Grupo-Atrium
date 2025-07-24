"""
Paquete principal del proyecto de Deep Learning.
Contiene todos los módulos para el sistema de recomendación de videos.
"""

# Importar configuración
try:
    from ..config import (
        BASE_DIR,
        DATA_DIR, 
        RAW_VIDEOS_DIR,
        FRAMES_DIR,
        DETECTIONS_DIR,
        CATEGORIAS,
        VIDEO_PARAMS,
        DOWNLOAD_PARAMS
    )
except:
    pass  # Para evitar errores si config no existe aún

# Importar módulos principales
from .detector import ObjectDetector
from .downloader import descargar_categorias, descargar_videos
from .frame_extractor import process_all_videos, extract_frames_ffmpeg_simple

# Definir qué se exporta con "from src import *"
__all__ = [
    'ObjectDetector',
    'descargar_categorias',
    'descargar_videos',
    'process_all_videos',
    'extract_frames_ffmpeg_simple'
]

# Versión del paquete
__version__ = '1.0.0'
__author__ = 'Grupo Atrium'

print("📦 Paquete Deep Learning cargado correctamente")