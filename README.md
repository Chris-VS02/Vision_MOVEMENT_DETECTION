# Detección de Movimiento en Video con OpenCV

Este proyecto implementa un sistema de detección de movimiento en video utilizando técnicas de visión por computadora con la biblioteca OpenCV en Python.
El sistema permite identificar y marcar regiones con movimiento en secuencias de video, utilizando el método de sustracción de fondo y detección de contornos.

## 📌 Características

- Carga y procesamiento de video desde archivo (`.mp4`, `.avi`, etc.).
- Conversión de fotogramas a escala de grises para mejorar el rendimiento.
- Aplicación de desenfoque gaussiano para reducir el ruido.
- Comparación de fotogramas consecutivos para detectar diferencias.
- Umbralización binaria para destacar las regiones en movimiento.
- Detección de contornos y filtrado por área para evitar falsos positivos.
- Visualización en tiempo real del video con los objetos en movimiento encerrados en rectángulos.

## 🧠 Tecnologías utilizadas

- Python 3.x
- OpenCV (`cv2`)
- Jupyter Notebook

## 🚀 Ejecución

1. Asegúrate de tener Python y OpenCV instalados. Puedes instalar OpenCV con:

   ```bash
   pip install opencv-python

## RESULTADOS
Flujo optico
https://github.com/user-attachments/assets/8763057c-441c-4c17-b697-ed264b08c590
