"""
Sistema de recomendación de vídeos.
Usa detecciones de objetos para encontrar vídeos similares de forma eficiente.
"""
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
import os
from collections import defaultdict
import pickle
import time

class VideoRecommender:
    """
    Recomendador de vídeos basado en similitud de contenido visual.
    """
    def __init__(self, detections_path="data/detections"):
        """Inicializa el recomendador."""
        self.detections_path = Path(detections_path)
        self.video_vectors = {}
        self.video_metadata = {}
        self.feature_names = []
        self.knn_model = None
        self.vectors_matrix = None
        self.video_names = []
        self.use_segmentation = False

    def _crear_vector_video_optimizado(self, df_detecciones):
        """
        Crea vector para un video.
        """
        if df_detecciones.empty:
            return {}
        # Vectorización usando pandas (más rápida)
        if 'area_porcentaje' in df_detecciones.columns:
            # Bonus: ponderar por área y confianza
            df_detecciones['peso'] = df_detecciones['confianza'] * (df_detecciones['area_porcentaje'] / 100.0)
        else:
            # Básico: solo confianza
            df_detecciones['peso'] = df_detecciones['confianza']
        # Agrupar por clase y sumar pesos
        vector_series = df_detecciones.groupby('clase')['peso'].sum()
        # Normalizar por frames
        total_frames = df_detecciones['frame'].nunique()
        if total_frames > 0:
            vector_series = vector_series / total_frames
        return vector_series.to_dict()

    def procesar_detecciones_optimizado(self, max_videos_per_category=5):
        """
        Procesa detecciones.
        Args:
            max_videos_per_category (int): Máximo de videos por categoría
        """
        print("PROCESANDO DETECCIONES")
        print("=" * 45)
        # Determinar qué tipo de detecciones usar
        segmentation_path = Path("data/detections_segmentacion")
        if segmentation_path.exists() and any(segmentation_path.iterdir()):
            detections_base = segmentation_path
            self.use_segmentation = True
            print("Usando detecciones de segmentación (bonus)")
        else:
            detections_base = self.detections_path
            print("Usando detecciones básicas")
            
        # Procesar videos con límite para eficiencia
        videos_procesados = 0
        for categoria in detections_base.iterdir():
            if not categoria.is_dir():
                continue
            print(f"\n{categoria.name}:")
            csv_files = list(categoria.glob("*_detections.csv"))[:max_videos_per_category]
            for csv_file in csv_files:
                try:
                    # Lectura optimizada
                    df = pd.read_csv(csv_file, dtype={'frame': 'int32'})
                    # Crear vector
                    video_vector = self._crear_vector_video_optimizado(df)
                    # Metadata
                    # Corrección: Extraer correctamente el nombre del video del archivo CSV
                    video_name = csv_file.stem.replace('_detections', '').replace('_segmentation', '')
                    full_video_name = f"{categoria.name}/{video_name}"
                    
                    # Depuración: Mostrar el nombre que se está almacenando
                    # print(f"Almacenando video: '{full_video_name}'")
                    
                    self.video_vectors[full_video_name] = video_vector
                    self.video_metadata[full_video_name] = {
                        'categoria': categoria.name,
                        'objetos': len(df)
                    }
                    videos_procesados += 1
                    if videos_procesados % 10 == 0:
                        print(f"{videos_procesados} videos procesados...")
                except Exception as e:
                    print(f"{csv_file.name}: {str(e)[:50]}")
                    continue
        print(f"\nTotal procesados: {videos_procesados}")
        return videos_procesados > 0

    def crear_matriz_optimizada(self, min_class_frequency=0.05):
        """
        Crea matriz eliminando clases poco frecuentes.
        Args:
            min_class_frequency (float): Frecuencia mínima para mantener clase
        """
        print("\nCREANDO MATRIZ OPTIMIZADA")
        print("=" * 35)
        if not self.video_vectors:
            return False
            
        # Obtener todas las clases
        all_classes = set()
        for vector in self.video_vectors.values():
            all_classes.update(vector.keys())
        print(f"Clases totales: {len(all_classes)}")
        
        # Filtrar clases poco frecuentes (optimización)
        if len(all_classes) > 500:  # Solo si hay muchas clases
            class_frequency = defaultdict(int)
            total_videos = len(self.video_vectors)
            for vector in self.video_vectors.values():
                for clase in vector.keys():
                    class_frequency[clase] += 1
            # Mantener clases que aparecen en al menos X% de videos
            min_videos = max(2, int(total_videos * min_class_frequency))
            filtered_classes = {clase for clase, freq in class_frequency.items() 
                              if freq >= min_videos}
            self.feature_names = sorted(list(filtered_classes))
            print(f"Clases filtradas: {len(self.feature_names)} (eliminadas: {len(all_classes) - len(self.feature_names)})")
        else:
            self.feature_names = sorted(list(all_classes))
            print(f"Clases mantenidas: {len(self.feature_names)}")
            
        # Crear matriz optimizada
        self.video_names = list(self.video_vectors.keys())
        self.vectors_matrix = np.zeros((len(self.video_names), len(self.feature_names)), dtype=np.float32)
        
        # Vectorización numpy
        class_to_idx = {clase: idx for idx, clase in enumerate(self.feature_names)}
        for i, video_name in enumerate(self.video_names):
            vector = self.video_vectors[video_name]
            for clase, valor in vector.items():
                if clase in class_to_idx:
                    self.vectors_matrix[i, class_to_idx[clase]] = valor
                    
        print(f"Matriz final: {self.vectors_matrix.shape}")
        return True

    def entrenar_knn_optimizado(self):
        """Entrena KNN optimizado."""
        print("\nENTRENANDO KNN OPTIMIZADO")
        print("=" * 30)
        if self.vectors_matrix is None or self.vectors_matrix.shape[0] == 0:
            print("No hay datos para entrenar")
            return False
            
        try:
            # KNN optimizado
            self.knn_model = NearestNeighbors(
                n_neighbors=min(11, len(self.video_names)),
                metric='cosine',
                algorithm='auto',  # Auto-optimización
                n_jobs=-1  # Usar todos los cores
            )
            self.knn_model.fit(self.vectors_matrix)
            print("KNN entrenado (multicore)")
            return True
        except Exception as e:
            print(f"Error entrenando KNN: {e}")
            return False

    def encontrar_similares_optimizado(self, video_name, top_k=10):
        """
        Encuentra similares.
        """
        if self.knn_model is None:
            return []
            
        try:
            # Búsqueda rápida
            # Corrección: Normalizar el nombre de entrada para evitar problemas de espacios
            video_name = video_name.strip()
            
            # Corrección: Verificar si el video existe mostrando nombres similares si no se encuentra
            if video_name not in self.video_names:
                print(f"Video no encontrado: '{video_name}'")
                # Mostrar videos que comienzan igual para ayudar al usuario
                similares = [name for name in self.video_names if name.startswith(video_name.split('/')[0] if '/' in video_name else video_name)]
                if similares:
                    print("Videos similares disponibles:")
                    for i, name in enumerate(similares[:5]):
                        print(f"   {i+1}. {name}")
                else:
                    print("Usa 'list' para ver todos los videos disponibles")
                return []
                
            video_idx = self.video_names.index(video_name)
            distances, indices = self.knn_model.kneighbors(
                [self.vectors_matrix[video_idx]], 
                n_neighbors=min(top_k + 5, len(self.video_names))  # Buscar más para filtrar
            )
            
            # Procesar resultados
            resultados = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx != video_idx and dist < 0.95:  # Filtrar muy diferentes
                    similitud = max(0, 1 - dist)
                    resultados.append((self.video_names[idx], float(similitud)))
                    if len(resultados) >= top_k:
                        break
            return sorted(resultados, key=lambda x: x[1], reverse=True)
        except ValueError as e:
            print(f"Video no encontrado: {video_name}")
            print(f"Error: {e}")
            return []
        except Exception as e:
            print(f"Error buscando similares: {e}")
            return []

    def mostrar_resultados_interactivos(self):
        """Muestra resultados de forma interactiva"""
        if not self.video_names:
            print("No hay videos procesados")
            return
            
        print("\nSISTEMA DE RECOMENDACIÓN ACTIVO")
        print("=" * 40)
        print(f"Videos disponibles: {len(self.video_names)}")
        print(f"Clases utilizadas: {len(self.feature_names)}")
        print("\nComandos:")
        print("   - 'list': Ver videos disponibles")
        print("   - 'recommend VIDEO_NAME': Recomendar similares")
        print("   - 'quit': Salir")
        
        while True:
            try:
                comando = input("\nComando: ").strip()
                # No convertir a minúsculas ya que los nombres de videos pueden tener mayúsculas
                if comando == 'quit':
                    break
                elif comando == 'list':
                    print(f"\nVideos disponibles ({len(self.video_names)}):")
                    for i, video in enumerate(self.video_names[:20]):  # Mostrar solo 20
                        categoria = self.video_metadata[video]['categoria']
                        print(f"   {i+1:2d}. {video} ({categoria})")
                    if len(self.video_names) > 20:
                        print(f"   ... y {len(self.video_names) - 20} más")
                elif comando.startswith('recommend '):
                    # Extraer el nombre del video manteniendo el caso original
                    video_input = comando[10:]  # No usar strip().lower()
                    video_input = video_input.strip()  # Solo quitar espacios innecesarios
                    
                    if video_input in self.video_names:
                        print(f"\nBuscando similares a: {video_input}")
                        similares = self.encontrar_similares_optimizado(video_input, top_k=10)
                        if similares:
                            print("TOP 10 RECOMENDACIONES:")
                            for i, (video, sim) in enumerate(similares, 1):
                                categoria = self.video_metadata[video]['categoria']
                                print(f"   {i:2d}. {video} ({categoria}) [{sim:.3f}]")
                        else:
                            print("No se encontraron recomendaciones")
                    else:
                        print("Video no encontrado. Usa 'list' para ver disponibles")
                else:
                    print("Comando no reconocido")
            except KeyboardInterrupt:
                print("\n¡Hasta luego!")
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    """
    Sistema de recomendación de vídeos.
    """
    print("RECOMENDADOR DE VÍDEOS")
    print("=" * 42)
    print("Optimizaciones implementadas:")
    print("  • Procesamiento vectorizado")
    print("  • Filtrado de clases poco relevantes") 
    print("  • KNN multicore")
    print("  • Memoria eficiente (float32)")
    print("  • Interfaz interactiva")
    print("=" * 42)
    
    start_time = time.time()
    
    # Crear y configurar recomendador
    recommender = VideoRecommender()
    
    # Procesar detecciones (optimizado)
    # Cambié max_videos_per_category a un número más alto para usar más datos
    if not recommender.procesar_detecciones_optimizado(max_videos_per_category=1000):
        print("No se pudieron procesar las detecciones")
        return
        
    # Crear matriz optimizada
    if not recommender.crear_matriz_optimizada(min_class_frequency=0.02):
        print("Error creando matriz de características")
        return
        
    # Entrenar modelo
    if not recommender.entrenar_knn_optimizado():
        print("Error entrenando modelo")
        return
        
    elapsed = time.time() - start_time
    print(f"\nSistema preparado en {elapsed:.1f} segundos")
    
    # Modo interactivo
    recommender.mostrar_resultados_interactivos()
    
    print("\n¡SISTEMA DE RECOMENDACIÓN COMPLETADO!")

if __name__ == "__main__":
    main()