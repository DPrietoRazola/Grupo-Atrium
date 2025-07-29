"""
Descarga de videos de YouTube.
"""

import yt_dlp
import os
from concurrent.futures import ThreadPoolExecutor
import threading
import time

# Bloqueo para threads
lock = threading.Lock()

def descargar_video_rapido(url_info, directorio_destino):
    """Descarga de videos rápidamente."""
    try:
        ydl_opts = {
            'outtmpl': os.path.join(directorio_destino, '%(id)s.%(ext)s'),
            'format': 'worst[ext=mp4]/worst',  # Calidad mínima
            'ignoreerrors': True,
            'quiet': True,  # Silencioso
            'nocheckcertificate': True,
            'ratelimit': 512*1024,  # Aumentar velocidad a 512 KB/s
            'merge_output_format': 'mp4',
            'noplaylist': True,
            'nooverwrites': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url_info['url']])
            
        with lock:
            print(f"{url_info['id']}")
            
    except Exception as e:
        with lock:
            print(f"{url_info['id'][:10]}...")

def buscar_y_descargar_rapido(consulta, directorio_destino, cantidad=100):
    """Busca y descarga videos en paralelo (optimizado)"""
    
    print(f"Buscando {cantidad} videos de: {consulta}")
    
    # Buscar videos primero
    search_opts = {
        'quiet': True,
        'ignoreerrors': True,
        'extract_flat': True,  # Solo metadatos
    }
    
    urls_info = []
    with yt_dlp.YoutubeDL(search_opts) as ydl:
        search_query = f"ytsearch{cantidad}:{consulta}"
        result = ydl.extract_info(search_query, download=False)
        
        if 'entries' in result:
            for entry in result['entries'][:cantidad]:
                if entry and 'id' in entry and 'url' in entry:
                    urls_info.append({
                        'id': entry['id'],
                        'url': entry['url']
                    })
    
    print(f"Descargando {len(urls_info)} videos en paralelo (8 simultáneos)...")
    
    # Descargar en paralelo
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for url_info in urls_info:
            future = executor.submit(descargar_video_rapido, url_info, directorio_destino)
            futures.append(future)
        
        # Esperar a que terminen
        for future in futures:
            future.result()

def descargar_todas_categorias_rapido():
    """Descarga todas las categorías de videos (optimizada)"""
    
    categorias = {
        "coches": "car crash compilation short",
        "mascotas": "funny cat compilation short", 
        "comida": "fast food commercial short"
    }
    
    directorio_base = "data/raw_videos"
    os.makedirs(directorio_base, exist_ok=True)
    
    print("DESCARGA DE VIDEOS DE YOUTUBE")
    print("=" * 55)
    
    start_time = time.time()
    
    for categoria, consulta in categorias.items():
        directorio_categoria = os.path.join(directorio_base, categoria)
        os.makedirs(directorio_categoria, exist_ok=True)
        print(f"\n{categoria.upper()} ({consulta}):")
        
        buscar_y_descargar_rapido(consulta, directorio_categoria, cantidad=100)  # Reducir a 100 por ahora
        
        time.sleep(1)  # Pequeña pausa
    
    total_time = time.time() - start_time
    print("\n" + "=" * 55)
    print(f"¡TODO LISTO! Tiempo total: {total_time/60:.1f} minutos")

if __name__ == "__main__":
    descargar_todas_categorias_rapido()
