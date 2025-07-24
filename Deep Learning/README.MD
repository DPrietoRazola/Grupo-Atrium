# 🎬 Sistema de Recomendación de Vídeos basado en Contenido

Proyecto de Deep Learning que recomienda vídeos similares basándose en los objetos que aparecen en los fotogramas.

## 📋 Descripción del Proyecto

Este sistema recomienda vídeos similares analizando los objetos presentes en los fotogramas. La premisa es que dos vídeos son más similares cuantos más objetos similares contienen.

## 📁 Estructura del Proyecto

Deep Learning/
├── config.py # Configuración del proyecto
├── src/ # Código fuente
│ ├── init .py # Paquete Python
│ ├── downloader.py # Descarga de vídeos de YouTube
│ ├── frame_extractor.py # Extracción de fotogramas
│ ├── detector.py # Detección de objetos
│ └── ver_resultados.py # Verificación de resultados
├── data/ # Datos del proyecto
│ ├── raw_videos/ # Vídeos originales descargados
│ ├── frames/ # Fotogramas extraídos
│ └── detections/ # Resultados de detección
