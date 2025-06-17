import cv2
import numpy as np
import os
import time
import pickle
import matplotlib.pyplot as plt
from tracker import CentroidTracker  # Asumiendo que lo guardaste en tracker.py


def create_background_subtractor(history=500, var_threshold=16, detect_shadows=True):
    """Crea un sustractor de fondo MOG2 con los parámetros indicados."""
    return cv2.createBackgroundSubtractorMOG2(
        history=history,
        varThreshold=var_threshold,
        detectShadows=detect_shadows
    )


def show_frame_visualization(frame, fgmask, bg_model, frame_count, fps, detect_shadows):
    """Muestra una visualización de los resultados cada cierto intervalo."""
    current_time = frame_count / fps
    minutes = int(current_time // 60)
    seconds = int(current_time % 60)

    plt.figure(figsize=(16, 10))

    plt.subplot(221)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title(f'Original - Tiempo: {minutes:02d}:{seconds:02d}')
    plt.axis('off')

    plt.subplot(222)
    if bg_model is not None:
        plt.imshow(cv2.cvtColor(bg_model, cv2.COLOR_BGR2RGB))
        plt.title('Modelo de Fondo')
    else:
        plt.title('Modelo de Fondo no disponible')
    plt.axis('off')

    plt.subplot(223)
    plt.imshow(fgmask, cmap='gray')
    plt.title('Máscara con sombras')
    plt.axis('off')

    plt.subplot(224)
    if detect_shadows:
        binary_mask = np.where(fgmask == 255, 255, 0).astype(np.uint8)
    else:
        binary_mask = fgmask
    plt.imshow(binary_mask, cmap='gray')
    plt.title('Máscara Binaria sin sombras')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def save_background_model(model_path, bg_model, parameters):
    """Guarda el modelo de fondo y parámetros en un archivo pickle."""
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    data = {'bg_model': bg_model, 'parameters': parameters}
    with open(model_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"✅ Modelo guardado en: {model_path}")


def load_background_model(model_path):
    """Carga el modelo de fondo desde un archivo."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo: {model_path}")
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def background_subtraction_mog2(video_path, duration=30, display_interval=30,
                                history=500, var_threshold=16, detect_shadows=True,
                                save_model=False, model_path=None, load_model=False):
    """Realiza sustracción de fondo sobre un video usando MOG2."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"No se encontró el video: {video_path}")

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    model_path = model_path or f"bg_model_{video_name}_h{history}_vt{var_threshold}.pkl"

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    max_frames = int(fps * duration)

    print(f"🔄 Procesando {max_frames} frames (~{duration}s a {fps:.2f} FPS)...")

    fgbg = create_background_subtractor(history, var_threshold, detect_shadows)

    if load_model:
        try:
            model_data = load_background_model(model_path)
            bg_model = model_data['bg_model']
            print(f"✅ Modelo cargado desde: {model_path}")
        except Exception as e:
            print(f"⚠️ Error al cargar modelo: {e}. Usando modelo nuevo.")
            bg_model = None
    else:
        bg_model = None

    frame_count = 0
    start_time = time.time()

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            print("📌 Fin del video.")
            break

        fgmask = fgbg.apply(frame)
        bg_model = fgbg.getBackgroundImage()

        if frame_count % display_interval == 0:
            show_frame_visualization(frame, fgmask, bg_model, frame_count, fps, detect_shadows)
            elapsed = time.time() - start_time
            print(f"🧮 Frame {frame_count}/{max_frames} | Tiempo: {elapsed:.2f}s | Velocidad: {frame_count/elapsed:.2f} FPS")

        frame_count += 1

    cap.release()

    if save_model and bg_model is not None:
        save_background_model(model_path, bg_model, {
            'history': history,
            'var_threshold': var_threshold,
            'detect_shadows': detect_shadows
        })

    print(f"\n✅ Proceso completado: {frame_count} frames en {time.time() - start_time:.2f}s")
    return bg_model


def apply_background_model(video_path, model_path, duration=30, display_interval=30):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

    model_data = load_background_model(model_path)
    parameters = model_data['parameters']
    fgbg = create_background_subtractor(**parameters)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    max_frames = int(fps * duration)

    tracker = CentroidTracker()

    frame_count = 0
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        fgmask = fgbg.apply(frame)
        refined_mask = refine_mask(fgmask)
        features = extract_features(refined_mask, frame)

        # Extrae centroides
        input_centroids = np.array([f['centroid'] for f in features], dtype="int")
        objects = tracker.update(input_centroids)

        if frame_count % display_interval == 0:
            result = frame.copy()

            # Dibujar bounding boxes e IDs
            for obj, feature in zip(objects.items(), features):
                object_id, centroid = obj
                x, y, w, h = feature['bounding_box']
                cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(result, f"ID {object_id}", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.circle(result, centroid, 4, (0, 0, 255), -1)

            # Mostrar resultados
            plt.figure(figsize=(14, 6))
            plt.subplot(131)
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.title('Original')
            plt.axis('off')

            plt.subplot(132)
            plt.imshow(refined_mask, cmap='gray')
            plt.title('Máscara Refinada')
            plt.axis('off')

            plt.subplot(133)
            plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            plt.title('Seguimiento de Objetos')
            plt.axis('off')

            plt.tight_layout()
            plt.show()

        frame_count += 1

    cap.release()
    print("✅ Seguimiento completado.")
    
def refine_mask(fgmask, shadow_val=127, min_area=500):
    """Procesa la máscara para eliminar sombras, reducir ruido y unir objetos."""
    # Eliminar sombras (valores 127) → quedarnos con objetos reales
    binary_mask = np.where(fgmask == 255, 255, 0).astype(np.uint8)

    # Operaciones morfológicas: apertura (ruido), cierre (agujeros), dilatación (unión)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
    clean = cv2.dilate(clean, kernel, iterations=1)

    # Filtrado por área mínima
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(clean)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(final_mask, [cnt], -1, 255, -1)

    return final_mask


def extract_features(mask, frame):
    """Extrae características de objetos detectados en la máscara."""
    features = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:  # ignorar objetos muy pequeños
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        cx = x + w // 2
        cy = y + h // 2

        obj_features = {
            'area': area,
            'centroid': (cx, cy),
            'bounding_box': (x, y, w, h),
            'aspect_ratio': aspect_ratio
        }

        features.append(obj_features)
    return features

