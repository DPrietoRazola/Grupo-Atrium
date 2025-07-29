"""
Detector de objetos basado en clasificador pre-entrenado.
Implementa la detección de objetos en fotogramas.
"""

import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import os
import pandas as pd
from pathlib import Path
import time

class ObjectDetector:
    """
    Detector de objetos usando ResNet50 pre-entrenado en ImageNet (1000 clases).
    """
    
    def __init__(self):
        """
        Inicializa el detector con modelo pre-entrenado de ImageNet.
        """
        print("Cargando modelo ResNet50 pre-entrenado...")
        # Cargar modelo pre-entrenado con 1000 clases de ImageNet
        self.weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=self.weights)
        self.model.eval()  # Modo evaluación
        
        # Transformaciones estándar para ImageNet
        self.preprocess = transforms.Compose([
            transforms.Resize(256),                    # Redimensionar a 256
            transforms.CenterCrop(224),                # Recortar al centro 224x224
            transforms.ToTensor(),                     # Convertir a tensor
            transforms.Normalize(                       # Normalizar como ImageNet
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        # Etiquetas de las 1000 clases de ImageNet
        self.labels = self.weights.meta["categories"]
        print("Modelo cargado correctamente")
    
    def inferencia_video(self, directorio_fotogramas):
        """
        Realiza inferencia en todos los fotogramas de un video.
        
        Args:
            directorio_fotogramas (str): Ruta al directorio con fotogramas ordenados
            
        Returns:
            pd.DataFrame: DataFrame con columnas ['frame', 'clase', 'confianza']
        """
        resultados = []
        
        # Obtener y ordenar los archivos de fotogramas
        frame_files = [f for f in os.listdir(directorio_fotogramas) 
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Ordenar numéricamente por el número del frame
        frame_files.sort(key=lambda x: int(x.split('.')[0]))
        
        print(f"Procesando {len(frame_files)} fotogramas...")
        
        for frame_file in frame_files:
            try:
                # Extraer número de frame
                frame_number = int(frame_file.split('.')[0])
                
                # Cargar y preprocesar imagen
                frame_path = os.path.join(directorio_fotogramas, frame_file)
                img = Image.open(frame_path).convert("RGB")
                img_t = self.preprocess(img)
                batch_t = torch.unsqueeze(img_t, 0)
                
                # Inferencia con el modelo
                with torch.no_grad():
                    output = self.model(batch_t)
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                    confidence, predicted_idx = torch.max(probabilities, 0)
                
                # Obtener clase predicha y confianza
                predicted_class = self.labels[predicted_idx.item()]
                confidence_value = confidence.item()
                
                # Guardar resultado
                resultados.append({
                    'frame': frame_number,
                    'clase': predicted_class,
                    'confianza': confidence_value
                })
                
            except Exception as e:
                print(f"Error procesando {frame_file}: {e}")
                continue
        
        # Crear DataFrame con los resultados
        df_resultados = pd.DataFrame(resultados)
        return df_resultados
    
    def procesar_todos_videos(self):
        """
        Procesa todos los videos y guarda resultados en CSV.
        """
        base_path = Path("data/frames")
        detections_path = Path("data/detections")
        detections_path.mkdir(exist_ok=True)
        
        total_videos = 0
        processed_videos = 0
        
        print("INICIANDO DETECCIÓN DE OBJETOS EN TODOS LOS VIDEOS")
        print("=" * 60)
        
        # Recorrer todas las categorías
        for categoria in ["coches", "mascotas", "comida"]:
            categoria_path = base_path / categoria
            if not categoria_path.exists():
                print(f"Categoría no encontrada: {categoria}")
                continue
            
            # Crear directorio de detecciones para esta categoría
            categoria_detections = detections_path / categoria
            categoria_detections.mkdir(exist_ok=True)
            
            # Recorrer todos los videos de la categoría
            video_folders = [f for f in categoria_path.iterdir() if f.is_dir()]
            total_videos += len(video_folders)
            
            print(f"\nCategoría: {categoria} ({len(video_folders)} videos)")
            
            for video_folder in video_folders:
                try:
                    print(f"Procesando: {video_folder.name}")
                    
                    # Detectar objetos
                    df_resultados = self.inferencia_video(str(video_folder))
                    
                    # Guardar resultados
                    csv_path = categoria_detections / f"{video_folder.name}_detections.csv"
                    df_resultados.to_csv(csv_path, index=False)
                    
                    print(f"Guardado: {csv_path.name}")
                    processed_videos += 1
                    
                except Exception as e:
                    print(f"Error en {video_folder.name}: {e}")
                    continue
        
        print("\n" + "=" * 60)
        print(f"DETECCIÓN COMPLETADA!")
        print(f"Videos procesados: {processed_videos}/{total_videos}")
        print(f"Resultados guardados en: data/detections/")
        print("Siguiente paso: Análisis de resultados")

def main():
    """
    Ejecuta el detector completo para todos los videos.
    """
    print("DETECTOR DE OBJETOS")
    print("=" * 50)
    print("Requisitos cumplidos:")
    print("  • Red neuronal convolucional pre-entrenada (ResNet50)")
    print("  • 1000 clases de ImageNet")
    print("  • DataFrame con frame, clase, confianza")
    print("  • Procesa cada directorio de fotogramas")
    print("=" * 50)
    
    # Crear detector
    detector = ObjectDetector()
    
    # Procesar todos los videos
    start_time = time.time()
    detector.procesar_todos_videos()
    elapsed_time = time.time() - start_time
    
    print(f"\nTiempo total: {elapsed_time/60:.1f} minutos")

if __name__ == "__main__":
    main()