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


## RESULTADOS
### Flujo optico
![OpticalFlow](https://github.com/user-attachments/assets/684b1580-6587-436a-9659-a20f2ac6bfba)


### Redes Neuronales
![NN](https://github.com/user-attachments/assets/a7453741-21a9-40ca-886d-36edb3eaf2b4)

