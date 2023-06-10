from kivy.uix.screenmanager import Screen
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.core.audio import SoundLoader
import cv2
import dlib
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from imutils import face_utils
from scipy.spatial import distance as dist
import threading

class DetectionScreen(Screen):
    def initialize(self):
        threading.Thread(target=self.initialize_resources).start()
        Clock.schedule_interval(self.update, 0.05)

    def initialize_resources(self):
        self.capture = cv2.VideoCapture(0)

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("Predictors/shape_predictor_68_face_landmarks.dat")

        self.update_event = None
        self.draw_landmarks = True
        
        # Initialisierung der Werte für die Blinzeldetektion
        self.count_frame = 0
        self.blink_thresh = 0.18
        self.succ_frame = 1

        # Initialisierung der Liste der Frames für die PERCLOS Berechnung
        self.list_of_eye_closure = []

        # Initialisierung der Liste der Frames für die blink-Threshold Berechnung
        self.list_of_EAR = []

        self.awake_ear_eyes_open = 0
        self.awake_perclos = 0.01

        self.count_last = -1
        self.cal_done = False

        self.blinks = 0

    def on_enter(self):
        self.start_camera()

    def on_leave(self):
        self.stop_camera()

    def start_camera(self):
        self.capture = cv2.VideoCapture(0)
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.update_event = Clock.schedule_interval(self.update, 1/self.fps)
        print(self.fps)

    def stop_camera(self):
        if hasattr(self, 'capture'):
            self.capture.release()
            self.capture = None
        if hasattr(self, 'update_event'):
            Clock.unschedule(self.update_event)
            self.update_event = None
        print(self.blinks)
        print(self.awake_ear_eyes_open)
        print(self.awake_perclos)

    def update(self, dt):
        # Eye landmarks
        (L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

        if hasattr(self, 'capture') and hasattr(self, 'fps') and self.manager.current == 'detection':
            ret, frame = self.capture.read()
            if ret:
                # Changing to Gray so that Dlib can process the frame
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                faces = self.detector(image)

                self.count_last +=1

                for rect in faces:
                    # Draw a rectangle around the detected face
                    x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()  # noqa: E501
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    # Get the landmarks for the face and draw them
                    shape = self.predictor(image, rect)
                    for i in range(shape.num_parts):
                        p = shape.part(i)
                        cv2.circle(frame, (p.x, p.y), 2, (0, 255, 0), -1)
                    
                    # converting the shape class directly
                    # to a list of (x,y) coordinates
                    shape = face_utils.shape_to_np(shape)
        
                    # parsing the landmarks list to extract
                    # lefteye and righteye landmarks--#
                    lefteye = shape[L_start: L_end]
                    righteye = shape[R_start:R_end]
                        
                    # Calculating the Eye Aspect ratio for the left and right eye
                    EAR_left = self.calculate_EAR(lefteye)
                        
                    EAR_right = self.calculate_EAR(righteye)

                    # Calculating the Average EAR for both eyes
                    avg_EAR = (EAR_right+EAR_left)/2

                    # Blink Detection Algorithm
                    blink, closed_eye = self.blink_detection(avg_EAR)

                    frame_length_perclos = 1000
                    frame_length_ear_list = 1000

                    # PERCLOS Calculation based on frames
                    perclos = self.calculate_perclos(closed_eye, frame_length_perclos)
                        
                    # AVG EAR for eyes open
                    self.get_list_of_ear(avg_EAR, frame_length_ear_list)
                    avg_ear_eyes_open_at_test = self.avg_ear_eyes_open()
                        
                    calibration = self.calibrate(
                        frame_length_perclos, frame_length_ear_list, 
                        perclos, avg_ear_eyes_open_at_test)

                    if self.cal_done:
                        string_perclos = "PERCLOS: " + str(perclos)
                        cv2.putText(frame, string_perclos, (30, 120),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)

                        feature_vector = self.feature_vector(
                        avg_ear_eyes_open_at_test, perclos)

                    else:
                        calibration = round(calibration, 2)*100
                        string_cal = "Calibration: " + str(calibration) + "%"
                        cv2.putText(frame, string_cal, (30, 120),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)

                    if blink == 1:
                        # Putting a text, that a blink is detected
                        cv2.putText(frame, 'Blink Detected', (30, 30),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)
                        self.blinks +=1
                        
                    if blink == 2:
                        # Putting a text, that driver might be sleeping
                        cv2.putText(frame, 'ALARM: Wake up!', (30, 30),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)
                        self.play_warning_sound()
                        self.blinks +=1

                        
                # Convert the image to a format that Kivy can use
                buf1 = cv2.flip(frame, 0)
                buf = buf1.tostring()
                image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), 
                                               colorfmt='bgr')
                image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

                self.ids.image_view.texture = image_texture

    def play_warning_sound(self):
        sound = SoundLoader.load('warning.ogg')
        if sound:
            sound.play()

    def blink_detection(self, avg_EAR):
        blink = 0
        if avg_EAR < self.blink_thresh:
            self.count_frame +=1
            eye_closed = True
        else:
            eye_closed = False
            if self.count_frame >= self.succ_frame:
                self.count_frame = 0
                blink = 1
            else:
                self.count_frame = 0
        
        if self.count_frame > self.fps/2:
            blink = 2
            
        return blink, eye_closed

    def calculate_perclos(self, closed_eye, length_of_frames):
        perclos = 0
        number_of_frames = len(self.list_of_eye_closure)

        if number_of_frames == length_of_frames:
            self.list_of_eye_closure.append(closed_eye)
            self.list_of_eye_closure.pop(0)
            
            frame_is_blink = self.list_of_eye_closure.count(True)
            perclos = frame_is_blink/number_of_frames

        elif number_of_frames < length_of_frames:
            self.list_of_eye_closure.append(closed_eye)

        else:
            print("Fehler, Liste zu lang")

        return perclos

    def calculate_EAR(self, eye):
        vertical1 = dist.euclidean(eye[1], eye[5])
        vertical2 = dist.euclidean(eye[2], eye[4])
                
        horizontal = dist.euclidean(eye[0], eye[3])
                
        EAR = (vertical1+vertical2)/(2*horizontal)
                    
        return EAR
    
    def get_list_of_ear(self, avg_ear, length):
        number_of_frames = len(self.list_of_EAR)

        if number_of_frames == length:
            self.list_of_EAR.append(avg_ear)
            self.list_of_EAR.pop(0)
            
        elif number_of_frames < length:
            self.list_of_EAR.append(avg_ear)

        else:
            pass
    
    def avg_ear_eyes_open(self):
        list_of_eyes_open = []
        avg_ear_eyes_open = -1

        if len(self.list_of_eye_closure) != len(self.list_of_EAR):
            print("Längen stimmen nicht überein")
        else:
            list_of_eyes_open = [self.list_of_EAR[i] for i in 
                                range(len(self.list_of_eye_closure)) 
                                if not self.list_of_eye_closure[i]]

            avg_ear_eyes_open = sum(list_of_eyes_open) / len(list_of_eyes_open)
        
        return avg_ear_eyes_open
    
    def calibrate(self, frame_length_perclos, frame_length_ear_list, perclos, ear_eyes_open):
        cal_perclos = False
        cal_ear = False
        calibrate_status = 0

        if frame_length_ear_list >= frame_length_perclos:
            calibrate_status = self.count_last/frame_length_ear_list
        else:
            calibrate_status = self.count_last/frame_length_perclos

        if self.count_last == frame_length_perclos:
            self.awake_perclos = perclos
            cal_perclos = True

        if self.count_last == frame_length_ear_list:
            self.awake_ear_eyes_open = ear_eyes_open
            cal_ear = True
        
        if cal_ear and cal_perclos:
            self.cal_done = True

        return calibrate_status
    
    def feature_vector(self, frame_ear_eyes_open, frame_perclos):
        diff_ear_eyes_open = self.awake_ear_eyes_open-frame_ear_eyes_open
        diff_perclos = self.awake_perclos-frame_perclos
        perclos = frame_perclos

        return [diff_ear_eyes_open, diff_perclos, perclos]
