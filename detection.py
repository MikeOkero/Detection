import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from playsound import playsound
from threading import Thread

class EyeStatus:
    OPEN = 0
    CLOSED = 1

def start_alarm(sound):
    """Play the alarm sound"""
    playsound(sound)

def detect_eyes(roi_color, model):
    eye = cv2.resize(roi_color, (80, 80))
    eye = eye.astype('float') / 255.0
    eye = img_to_array(eye)
    eye = np.expand_dims(eye, axis=0)
    pred = model.predict(eye)
    return np.argmax(pred)

# def detect_yawn(mouth_roi, yawn_model):
#     resized_mouth = cv2.resize(mouth_roi, (80,80))
#     preprocessed_mouth = preprocess_yawn_image(resized_mouth)
#     prediction = yawn_model.predict(preprocessed_mouth)
#     yawning_detected = prediction >= YOUR_YAWN_THRESHOLD
#     return yawning_detected

def main():
    classes = ['Open', 'Closed']
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    left_eye_cascade = cv2.CascadeClassifier("haarcascade_lefteye_2splits.xml")
    right_eye_cascade = cv2.CascadeClassifier("haarcascade_righteye_2splits.xml")
    cap = cv2.VideoCapture(0)
    model = load_model("best_model.h5")
    count = 0
    alarm_on = False
    alarm_sound = "alarm.mp3"
    status1 = EyeStatus.OPEN
    status2 = EyeStatus.OPEN

    while True:
        _, frame = cap.read()
        height = frame.shape[0]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            left_eye = left_eye_cascade.detectMultiScale(roi_gray)
            right_eye = right_eye_cascade.detectMultiScale(roi_gray)

            status1 = EyeStatus.OPEN
            status2 = EyeStatus.OPEN

            if left_eye:
                status1 = detect_eyes(roi_color[y:y+h1, x:x+w1], model)

            if right_eye:
                status2 = detect_eyes(roi_color[y:y+h2, x:x+w2], model)

            if status1 == EyeStatus.CLOSED and status2 == EyeStatus.CLOSED:
                count += 1
                cv2.putText(frame, "Eyes Closed, Frame count: " + str(count), (10, 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

                if count >= 10:
                    cv2.putText(frame, "Drowsiness Alert!!!", (100, height-20),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

                    if not alarm_on:
                        alarm_on = True
                        # play the alarm sound in a new thread
                        t = Thread(target=start_alarm, args=(alarm_sound,))
                        t.daemon = True
                        t.start()
            else:
                cv2.putText(frame, "Eyes Open", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
                count = 0
                alarm_on = False

        cv2.imshow("Driver Drowsiness Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
