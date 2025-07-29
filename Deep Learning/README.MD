# 🎬 Sistema de Recomendación de Vídeos basado en Contenido

Proyecto de Deep Learning que recomienda vídeos similares basándose en los objetos que aparecen en los fotogramas.

## 📋 Descripción del Proyecto

Este sistema recomienda vídeos similares analizando los objetos presentes en los fotogramas. La premisa es que dos vídeos son más similares cuantos más objetos similares contienen.

## Características Principales

- **Detección de Objetos:** Utiliza modelos pre-entrenados (como ResNet50 o Mask R-CNN) para detectar objetos en los fotogramas.
- **Extracción de Fotogramas:** Extrae fotogramas clave de los vídeos para procesamiento.
- **Recomendación Basada en Similitud:** Implementa un algoritmo K-Nearest Neighbors (KNN) optimizado para encontrar vídeos similares.
- **Interfaz Interactiva:** Permite a los usuarios buscar vídeos y recibir recomendaciones de manera sencilla.
