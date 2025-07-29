"""
Extractor de frames según el enunciado: procesa CADA VÍDEO
"""

import ffmpeg
import os

def extract_frames_ffmpeg_simple(video_path, output_folder, frame_rate=3, resolution=(224, 224)):
    """
    Extrae fotogramas de un video usando FFmpeg.
    """
    try:
        os.makedirs(output_folder, exist_ok=True)
        (
            ffmpeg
            .input(video_path)
            .filter('fps', fps=frame_rate)  # 3 frames por segundo
            .filter('scale', width=resolution[0], height=resolution[1])  # 224x224
            .output(os.path.join(output_folder, '%d.jpg'), format='image2', vframes=1000)
            .run(quiet=True, overwrite_output=True)
        )
        print(f"Frames extraídos: {os.path.basename(video_path)}")
        return True
    except Exception as e:
        print(f"Error {os.path.basename(video_path)}: {e}")
        return False

def process_all_videos():
    """
    Procesa CADA VÍDEO como pide el enunciado.
    """
    categories = ["coches", "mascotas", "comida"]
    input_base_dir = "data/raw_videos"
    output_base_dir = "data/frames"
    frame_rate = 3
    resolution = (224, 224)
    
    total_videos = 0
    processed_videos = 0
    
    print("PROCESANDO CADA VÍDEO según enunciado...")
    print("=" * 50)
    
    for category in categories:
        category_dir = os.path.join(input_base_dir, category)
        if not os.path.exists(category_dir):
            print(f"Categoría no encontrada: {category}")
            continue
            
        video_files = [f for f in os.listdir(category_dir) if f.endswith((".mp4", ".mkv", ".avi"))]
        total_videos += len(video_files)
        
        print(f"{category}: {len(video_files)} videos")
        
        for video_file in video_files:
            video_path = os.path.join(category_dir, video_file)
            video_name = os.path.splitext(video_file)[0]
            output_folder = os.path.join(output_base_dir, category, video_name)
            
            if extract_frames_ffmpeg_simple(video_path, output_folder, frame_rate, resolution):
                processed_videos += 1
    
    print("\n" + "=" * 50)
    print(f"RESULTADO FINAL:")
    print(f"Procesados: {processed_videos}/{total_videos} videos")
    print("Ahora ejecuta: python src/detector.py")

if __name__ == "__main__":
    process_all_videos()