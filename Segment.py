# Importamos las librerias
from ultralytics import YOLO
import cv2

# Leer nuestro modelo
model = YOLO("best.pt")

# Realizar VideoCaptura
cap = cv2.VideoCapture(0)

# Bucle
while True:
    # Leer nuestros fotogramas
    ret, frame = cap.read()
    if not ret:
        print("No se pudo capturar el fotograma de la cámara")
        break

    # Leemos resultados
    resultados = model.predict(frame, imgsz=640, conf=0.5)  # Reducido el umbral de confianza a 0.5

    # Imprimir resultados para depuración
    print(resultados)

    # Mostramos resultados
    anotaciones = resultados[0].plot()

    # Mostramos nuestros fotogramas
    cv2.imshow("DETECCION Y SEGMENTACION", anotaciones)

    # Cerrar nuestro programa
    if cv2.waitKey(1) == 27:  # Presionar Esc para salir
        break

cap.release()
cv2.destroyAllWindows()
