import cv2
import face_recognition
import pyttsx3

def initialize_text_to_speech():
    engine = pyttsx3.init()
    return engine

def load_known_faces(user_data):
    known_face_encodings = []
    known_face_names = []

    for user_id, user_name in user_data.items():
        known_image = face_recognition.load_image_file(f"known_faces/{user_id}.jpg")
        known_encoding = face_recognition.face_encodings(known_image)[0]
        known_face_encodings.append(known_encoding)
        known_face_names.append(user_name)

    return known_face_encodings, known_face_names

def recognize_faces(frame, known_face_encodings, known_face_names, engine):
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        output_message = f"Hello, {name}!"
        print(output_message)
        engine.say(output_message)
        engine.runAndWait()

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    return frame

def main():
    cap = cv2.VideoCapture(0)

    engine = initialize_text_to_speech()

    user_data = {
        "1": "Aniket",
        "2": "Gaurav",
        # Add more user IDs and names as needed
    }

    known_face_encodings, known_face_names = load_known_faces(user_data)

    while True:
        ret, frame = cap.read()

        frame = recognize_faces(frame, known_face_encodings, known_face_names, engine)

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
