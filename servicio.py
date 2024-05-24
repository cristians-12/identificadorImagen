import cv2
import pytesseract
import numpy as np

# Configura la ruta al ejecutable de Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Crisbau\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# Carga la imagen
img = cv2.imread('placa.jpg')

# Convierte la imagen a escala de grises
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Aplica un filtro gaussiano para reducir el ruido
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# Realiza la detección de bordes con Canny
edged = cv2.Canny(gray, 50, 150)

# Encuentra los contornos en la imagen
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Ordena los contornos por área y encuentra la región que podría ser la placa
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
screenCnt = None

for c in contours:
    # Aproxima el contorno a un rectángulo
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # Si el contorno tiene cuatro vértices, lo consideramos como la placa
    if len(approx) == 4:
        screenCnt = approx
        break

# Recorta y muestra la región de la placa
if screenCnt is not None:
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [screenCnt], -1, 255, -1)
    masked = cv2.bitwise_and(img, img, mask=mask)
    plate = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

    # Aplica OCR a la región de la placa
    text = pytesseract.image_to_string(plate, config='--psm 6')
    print("Texto reconocido en la placa:", text)

    # Dibuja el contorno de la placa en la imagen original
    cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 2)
    cv2.imshow("Placa", plate)

cv2.imshow("Imagen original", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# import cv2
# import pytesseract
# from PIL import Image

# # Configura la ruta al ejecutable de Tesseract OCR (puede variar según tu instalación)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Crisbau\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
# img = Image.open("placa.jpg")
# text = pytesseract.image_to_string(img)
# print(text)

# Abre la cámara
# video = cv2.VideoCapture('http://192.168.1.4:8080/video')

# while video.isOpened():
#     ret, frame = video.read()
#     if not ret:
#         break
    
#     # Convierte la imagen a escala de grises
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#     # Realiza un umbral adaptativo en la imagen
#     _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
#     # Utiliza Tesseract OCR para reconocer texto en la imagen
#     text = pytesseract.image_to_string(thresh, lang='spa')
    
#     # Muestra la imagen y el texto reconocido
#     cv2.imshow('Video de entrada', frame)
#     print("Texto reconocido:", text)
    
#     # Espera 100 milisegundos (0.1 segundos) entre frames
#     if cv2.waitKey(100) & 0xFF == ord('q'):
#         break

# video.release()
# cv2.destroyAllWindows()