import cv2

#Cargar la cascada (Los datos de entrenamiento)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#Read the input image(Leemos la imagen de entrada)
img = cv2.imread('face2.jpg')
#Convert into grayscale ( conversion a escala de grises)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#Detected faces (deteccion facial)
face = face_cascade.detectMultiScale(gray,1.1,4)
#Draw rectangle around the face (Dibujo identificador de caras)
for (x, y ,w,h)in face:
    cv2.rectangle(img,(x,y),(x+w, y+w), (255,0,0),2)
#Display the output (Mostramos el resultado)
cv2.imshow('img',img)
cv2.waitKey()
