import face_recognition # importing module
import cv2
import numpy as np

# Welcome to satyams application

capture_video = cv2.VideoCapture(0) # 0 = defult video source which is the computers webcam

Vanshi_image = face_recognition.load_image_file('female.jpeg') # these are the images
# which are being trained by the system. With these images, once the face is
# displayed the program is trained to check if the face matches what is in the "data base"
female_encoding = face_recognition.face_encodings(Vanshi_image) [0]
Ansh_image = face_recognition.load_image_file('male.jpeg') # similar principle applied to Ansh_image
male_encoding = face_recognition.face_encodings(Ansh_image) [0]
rekha_image = face_recognition.load_image_file('rekha.jpeg')
rekha_encoding = face_recognition.face_encodings(rekha_image) [0]
satyam_image = face_recognition.load_image_file('satyam.jpeg')
satyam_encoding = face_recognition.face_encodings(satyam_image) [0]


# researc hhow to add more pictures

# to add more faces the same principles as above can be applied again, by declaring a "name"_image variable
# and the encodings


known_face_encodings = [
    female_encoding,
    male_encoding,
    rekha_encoding,
    satyam_encoding
]

known_face_names = [
    "Vanshi",
    "Ansh",
    "Rekha",
    "satyam",
]
# voice control thing
# declaring the initial variables
face_location = []
face_name = []
face_encodings = []
process_frame = True

while True:
    ret, frame = capture_video.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    if process_frame:
        face_location = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_location)

        face_name = []
        for face_encoding in face_encodings:
            # we will now check if all faces match the known faces
            matches = face_recognition.compare_faces(known_face_encodings,face_encoding)
            name = "Unknown"
            # We can also use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_name.append(name)

            # how to add faces automaticaly

       # print(face_location)

    process_frame = not process_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_location, face_name):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
       # left = left*4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (300, 300, 300), 1)
            # passing specific values which will be displayed once a face is detected

    # producing the image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break # stopping the program from running without causing KeyboardInterrupt error by pressing "q"

capture_video.release()
cv2.destroyAllWindows()
