"""
Detector de objetos basado en segmentación semántica (Bonus).
Usa Mask R-CNN pre-entrenado en MSCOCO para detección detallada.
"""

import torch
import torchvision.transforms as transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from PIL import Image
import numpy as np
import pandas as pd
import os
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
import threading

# Bloqueo para threads
lock = threading.Lock()

class SemanticSegmentationDetector:
    """
    Detector de segmentación semántica optimizado usando Mask R-CNN.
    """
    
    def __init__(self, confidence_threshold=0.5, max_detections_per_frame=10):
        """
        Inicializa el detector optimizado.
        
        Args:
            confidence_threshold (float): Umbral mínimo de confianza
            max_detections_per_frame (int): Máximo de detecciones por frame
        """
        print("Cargando modelo Mask R-CNN optimizado...")
        
        # Cargar modelo pre-entrenado (modo evaluación)
        self.weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = maskrcnn_resnet50_fpn(weights=self.weights)
        self.model.eval()
        self.model = self.model.to('cpu')  # Mantener en CPU para estabilidad
        
        # Transformaciones optimizadas
        self.transform = self.weights.transforms()
        
        # Parámetros
        self.confidence_threshold = confidence_threshold
        self.max_detections_per_frame = max_detections_per_frame
        self.coco_labels = self.weights.meta["categories"]
        
        print(f"Modelo optimizado cargado ({len(self.coco_labels)} clases)")
    
    def calcular_area_porcentaje(self, mask):
        """
        Calcula área porcentual de forma vectorizada.
        """
        # Método vectorizado más eficiente
        mask_tensor = torch.from_numpy(mask)
        area_percentage = (torch.sum(mask_tensor > 0).item() / mask_tensor.numel()) * 100
        return round(float(area_percentage), 2)
    
    def detectar_objetos_frame_optimizado(self, frame_path, frame_number):
        """
        Detección optimizada de objetos en frame.
        """
        try:
            # Cargar y preprocesar imagen
            image = Image.open(frame_path).convert("RGB")
            
            # Transformar para modelo
            input_tensor = self.transform(image)
            input_batch = input_tensor.unsqueeze(0)
            
            # Inferencia
            with torch.no_grad():
                prediction = self.model(input_batch)[0]
            
            detecciones = []
            detection_count = 0
            
            # Procesar detecciones (limitado por rendimiento)
            for i in range(min(len(prediction['scores']), self.max_detections_per_frame)):
                score = prediction['scores'][i].item()
                
                # Filtrar por confianza
                if score >= self.confidence_threshold:
                    # Obtener clase
                    class_id = prediction['labels'][i].item()
                    class_name = self.coco_labels[class_id] if class_id < len(self.coco_labels) else f"unknown_{class_id}"
                    
                    # Obtener y procesar máscara
                    mask = prediction['masks'][i, 0].mul(255).byte().cpu().numpy()
                    area_percentage = self.calcular_area_porcentaje(mask)
                    
                    # Añadir detección
                    detecciones.append({
                        'frame': frame_number,
                        'clase': class_name,
                        'confianza': round(score, 4),
                        'area_porcentaje': area_percentage
                    })
                    
                    detection_count += 1
            
            with lock:
                if detecciones:
                    print(f"Frame {frame_number}: {detection_count} objetos detectados")
            
            return detecciones
            
        except Exception as e:
            with lock:
                print(f"Error frame {frame_number}: {str(e)[:50]}...")
            return []
    
    def procesar_video_paralelo(self, frames_directory, max_workers=2):
        """
        Procesa video con paralelización limitada para estabilidad.
        """
        # Obtener frames ordenados
        frame_files = [f for f in os.listdir(frames_directory) 
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
        frame_files.sort(key=lambda x: int(x.split('.')[0]))
        
        # Limitar frames para prueba rápida y estabilidad
        frame_files = frame_files[:50]  # Solo primeros 50 frames
        print(f"Procesando {len(frame_files)} frames (limitado para estabilidad)...")
        
        resultados = []
        
        # Procesamiento paralelo (máximo 2 workers para evitar sobrecarga)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for frame_file in frame_files:
                frame_number = int(frame_file.split('.')[0])
                frame_path = os.path.join(frames_directory, frame_file)
                
                future = executor.submit(
                    self.detectar_objetos_frame_optimizado, 
                    frame_path, 
                    frame_number
                )
                futures.append(future)
            
            # Recoger resultados
            for future in futures:
                detecciones = future.result()
                resultados.extend(detecciones)
        
        return pd.DataFrame(resultados)
    
    def procesar_muestra_videos(self):
        """
        Procesa muestra de videos optimizada.
        """
        base_path = Path("data/frames")
        detections_path = Path("data/detections_segmentacion")
        detections_path.mkdir(exist_ok=True)
        
        print("DETECCIÓN OPTIMIZADA CON SEGMENTACIÓN SEMÁNTICA")
        print("=" * 60)
        print(f"Configuración: confianza>{self.confidence_threshold}, max {self.max_detections_per_frame} objetos/frame")
        print("=" * 60)
        
        processed_videos = 0
        
        # Procesar muestra (2 videos por categoría para prueba)
        for categoria in ["coches", "mascotas", "comida"]:
            categoria_path = base_path / categoria
            if not categoria_path.exists():
                continue
            
            categoria_detections = detections_path / categoria
            categoria_detections.mkdir(exist_ok=True)
            
            # Obtener solo primeros 2 videos de cada categoría
            video_folders = [f for f in categoria_path.iterdir() if f.is_dir()][:2]
            
            if video_folders:
                print(f"\n{categoria.upper()}: {len(video_folders)} videos")
                
                for video_folder in video_folders:
                    try:
                        print(f"Procesando: {video_folder.name}")
                        start_time = time.time()
                        
                        # Procesar video
                        df_resultados = self.procesar_video_paralelo(str(video_folder))
                        
                        # Guardar resultados
                        csv_filename = f"{video_folder.name}_segmentation_detections.csv"
                        csv_path = categoria_detections / csv_filename
                        df_resultados.to_csv(csv_path, index=False)
                        
                        # Estadísticas
                        elapsed = time.time() - start_time
                        objetos = len(df_resultados)
                        frames = df_resultados['frame'].nunique() if len(df_resultados) > 0 else 0
                        
                        print(f"{csv_filename} ({objetos} objetos, {frames} frames, {elapsed:.1f}s)")
                        processed_videos += 1
                        
                    except Exception as e:
                        print(f"Error {video_folder.name}: {e}")
        
        print(f"\n¡MUESTRA OPTIMIZADA COMPLETADA! ({processed_videos} videos)")

def main():
    """
    Ejecuta detector optimizado para muestra de videos.
    """
    print("DETECTOR DE SEGMENTACIÓN OPTIMIZADO (BONUS)")
    print("=" * 55)
    print("MEJORAS IMPLEMENTADAS:")
    print("  • Procesamiento paralelo controlado")
    print("  • Límite de frames para estabilidad") 
    print("  • Gestión eficiente de memoria")
    print("  • Salida detallada y progresiva")
    print("  • Manejo robusto de errores")
    print("=" * 55)
    
    # Detector optimizado
    detector = SemanticSegmentationDetector(
        confidence_threshold=0.4,      # Más permisivo
        max_detections_per_frame=8     # Control de carga
    )
    
    # Procesar muestra
    start_total = time.time()
    detector.procesar_muestra_videos()
    elapsed_total = time.time() - start_total
    
    print(f"\nTiempo total: {elapsed_total/60:.1f} minutos")

if __name__ == "__main__":
    main()