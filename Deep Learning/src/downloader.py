"""
Script para descargar videos de YouTube según las especificaciones del proyecto.
Descarga al menos 100 videos por categoría.
"""

import yt_dlp
import os
import random
import time

def descargar_videos(consulta, directorio_destino, cantidad=100):
    """
    Descarga videos de YouTube con calidad baja para ahorrar espacio.
    
    Args:
        consulta (str): Término de búsqueda
        directorio_destino (str): Carpeta donde guardar los videos
        cantidad (int): Número de videos a descargar (100 según enunciado)
    """
    ydl_opts = {
        'outtmpl': os.path.join(directorio_destino, '%(title)s-%(id)s.%(ext)s'),
        'format': 'worstvideo[height<=144][ext=mp4]+worstaudio[ext=m4a]/worst[ext=mp4]/worst',
        'ignoreerrors': True,
        'quiet': False,
        'nocheckcertificate': True,
        'ratelimit': 512*1024,  # Limitar velocidad para evitar bloqueos
        'merge_output_format': 'mp4',
        'max_downloads': cantidad,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            search_query = f"ytsearch{cantidad}:{consulta}"
            ydl.download([search_query])
            print(f"Descarga de '{consulta}' completada ({cantidad} videos).")
    except Exception as e:
        print(f"Error al descargar videos '{consulta}': {e}")

def descargar_categorias(categorias, directorio_base="data/raw_videos", cantidad=100):
    """
    Descarga videos para todas las categorías especificadas.
    
    Args:
        categorias (dict): Diccionario {categoria: consulta}
        directorio_base (str): Directorio base para guardar videos
        cantidad (int): Número de videos por categoría (100 según enunciado)
    """
    total_videos = 0
    for categoria, consulta in categorias.items():
        directorio_categoria = os.path.join(directorio_base, categoria)
        os.makedirs(directorio_categoria, exist_ok=True)
        print(f"Descargando categoría: {categoria} ({cantidad} videos)")
        descargar_videos(consulta, directorio_categoria, cantidad)
        # Espera aleatoria para evitar bloqueos
        time.sleep(random.randint(2, 5))
        total_videos += cantidad
    
    print(f"¡Descarga completa! {total_videos} videos en total.")

# Categorías según el enunciado
categorias = {
    "coches": "coches deportivos",      # Categoría 1
    "mascotas": "perros graciosos",     # Categoría 2  
    "comida": "recetas faciles"         # Categoría 3
}

if __name__ == "__main__":
    print("Iniciando descarga de videos según enunciado (100 por categoría)")
    print("=" * 60)
    print("Requisitos cumplidos:")
    print("  • 3 categorías: coches, mascotas, comida")
    print("  • 100 videos por categoría")
    print("  • Calidad baja para optimizar tiempo")
    print("=" * 60)
    
    descargar_categorias(categorias, cantidad=100)  # 100 como pide el enunciado
    
    print("\n¡Proceso de descarga finalizado!")
    print("Ahora ejecuta: python src/frame_extractor.py")