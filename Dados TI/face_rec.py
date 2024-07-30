import face_recognition
import os


# load the image 201112611.jpg from the Fotos folder
image = face_recognition.load_image_file('Fotos/20112611.jpg')
# load the image image1718132236.jpg from the uploads folder
image2 = face_recognition.load_image_file('uploads/image1718132236.jpg')

#compare the faces
face_encodings = face_recognition.face_encodings(image)
face_encodings2 = face_recognition.face_encodings(image2)

if len(face_encodings) == 0:
    print("No face found in image")
else:
    print("Face found in image")

if len(face_encodings2) == 0:
    print("No face found in image2")
else:
    print("Face found in image2")

matches = face_recognition.compare_faces(face_encodings, face_encodings2[0], tolerance=0.6)

if True in matches:
    print("Match found")
else:
    print("No match found")

