"""
Configuración centralizada del proyecto
"""

import os
from pathlib import Path

# Rutas base
BASE_DIR = Path("/home/dpr/Documentos/GitHub/Grupo-Atrium/Deep Learning")
DATA_DIR = BASE_DIR / "data"
RAW_VIDEOS_DIR = DATA_DIR / "raw_videos"
FRAMES_DIR = DATA_DIR / "frames"
DETECTIONS_DIR = DATA_DIR / "detections"

# Categorías del proyecto
CATEGORIAS = ["coches", "mascotas", "comida"]

# Parámetros de procesamiento
VIDEO_PARAMS = {
    'frame_rate': 3,        # 3 FPS como pide el enunciado
    'resolution': (224, 224)  # Resolución 224x224
}

# Opciones de descarga
DOWNLOAD_PARAMS = {
    'videos_por_categoria': 100,
    'calidad': 'worst[ext=mp4]/worst',
    'ratelimit': 512 * 1024,  # Limitar velocidad
    'merge_output_format': 'mp4'
}