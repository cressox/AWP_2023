from kivy.uix.screenmanager import Screen
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
import cv2
from kivy.core.audio import SoundLoader
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.logger import Logger
from matplotlib import pyplot as plt
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist

class DetectionScreen(Screen):
    def __init__(self, **kwargs):
        super(DetectionScreen, self).__init__(**kwargs)
        layout = BoxLayout(orientation='vertical')

        button_to_main = Button(text='Go to Main', size_hint=(None, None), 
                                size=(100, 50), pos_hint={'x': 0, 'y': 1})
        button_to_main.bind(on_press=lambda x: self.set_screen('main'))
        layout.add_widget(button_to_main)

        self.image = Image()
        layout.add_widget(self.image)

        self.add_widget(layout)

        Clock.schedule_interval(self.update, 0.02)

        self.fps = 0

        self.capture = None
        self.update_event = None
        self.draw_landmarks = True

        # Loading the important packages from Mediapipe

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.denormalize_coordinates = self.mp_drawing._normalized_to_pixel_coordinates
        
        # Defining the Face Mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=False,         # Default=False
                    max_num_faces=1,                # Default=1
                    # Unlike the default mode, this gives us the landmarks of the 
                    # iris and eyes in addition to the facial landmarks (478 in total)
                    refine_landmarks=True,         # Default=False
                    min_detection_confidence=0.5,   # Default=0.5
                    min_tracking_confidence= 0.5,)
        
        #Initialisierung der Werte für die Blinzeldetektion
        self.count_frame = 0
        self.blink_thresh = 0.18
        self.succ_frame = 1

        #Initialisierung der Liste der Frames für die PERCLOS Berechnung
        self.list_of_eye_closure = []

        #Initialisierung der Liste der Frames für die blink-Threshold Berechnung
        self.list_of_EAR = []

        Logger.info("Mediapipe: 478 Landmarks are detected")

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
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        if self.update_event is not None:
            Clock.unschedule(self.update_event)
            self.update_event = None

    def update(self, dt):
        if self.capture is not None:
            ret, frame = self.capture.read()
            if ret:
                # Changing to RGB so that mediapipe can process the frame
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                image = np.ascontiguousarray(image)
                imgH, imgW, _ = image.shape
                
                # Generation of the face mesh
                results = self.face_mesh.process(image)

                # Processing of the landmarks
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        
                        # Select the 6 landmarks per eye 
                        # for the calculation of the eye aspect ratio
                        # The chosen 12 points:   P1,  P2,  P3,  P4,  P5,  P6
                        left_eye_idxs  =        [362, 385, 387, 263, 373, 380]
                        right_eye_idxs =        [33,  160, 158, 133, 153, 144]

                        eye_idxs = left_eye_idxs + right_eye_idxs

                        # Drawing the 6 landmarks per eye
                        for landmark_idx, landmark in enumerate(
                            face_landmarks.landmark):
                            if landmark_idx in eye_idxs:
                                pred_cord = self.denormalize_coordinates(
                                    landmark.x, landmark.y, imgW, imgH)
                                cv2.circle(image, pred_cord,3,(255, 255, 255), -1)

                        # Getting the coordinate points for left and right eye
                        coord_points_left = self.get_coord_points(
                            face_landmarks.landmark, left_eye_idxs, imgW, imgH)
                        
                        coord_points_right = self.get_coord_points(
                            face_landmarks.landmark, right_eye_idxs, imgW, imgH)
                        
                        #Calculating the Eye Aspect ratio for the left and right eye
                        EAR_left = self.calculate_EAR(coord_points_left)
                        
                        EAR_right = self.calculate_EAR(coord_points_right)

                        # Calculating the Average EAR for both eyes
                        avg_EAR = (EAR_right+EAR_left)/2

                        # Blink Detection Algorithm
                        blink, closed_eye = self.blink_detection(avg_EAR)

                        #PERCLOS Calculation based on frames
                        perclos = self.calculate_perclos(closed_eye, 1000)
                        perclos = round(perclos, 2)

                        #Putting Text, first loading percentage, then Perclos value
                        if perclos > 1:
                            string_perclos = "Loading: " + str(perclos) + "%"
                            cv2.putText(image, string_perclos, (30, 120),
                            cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)
                        else:
                            string_perclos = "PERCLOS: " + str(perclos)
                            cv2.putText(image, string_perclos, (30, 120),
                            cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)
                        
                        if blink == 1:
                            # Putting a text, that a blink is detected
                            cv2.putText(image, 'Blink Detected', (30, 30),
                            cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)
                        
                        if blink == 2:
                            # Putting a text, that driver might be sleeping
                            cv2.putText(image, 'ALARM: Wake up!', (30, 30),
                            cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)
                            self.play_warning_sound()

                        
                buf1 = cv2.flip(image, 0)
                buf = buf1.tostring()
                image_texture = Texture.create(size=(image.shape[1], image.shape[0]), 
                colorfmt='rgb')
                image_texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')

                self.image.texture = image_texture

    def play_warning_sound(self):
        sound = SoundLoader.load('warning.ogg')
        if sound:
            sound.play()

    def set_screen(self, screen_name):
        self.manager.current = screen_name 

    def blink_detection(self, avg_EAR: float):
        """Calculates the blink behavior based on the EAR value. 
        If the threshold (how far the eyes are closed)
        is not reached , blinking is detected.

        Args:
            avg_EAR (float): Transfer of the EAR value 
            over both eyes for a recorded frame

        Returns:
            blink (Int): Indicates whether a blink was just detected, 
            0 = No Blink, 1 = Blink, 2 = Sleep, too long period of time closed eyes
            eye_closed (Bool): Indicates whether in the inputframe
            the eye is cloed or not
        """
        blink = 0
        # Counting the frames when there is a blink
        if avg_EAR < self.blink_thresh:
           self.count_frame +=1
           eye_closed = True
        else:
            eye_closed = False
            # The blink is done, if the counting stops 
            # and the EAR is bigger than the blink threshold
            if self.count_frame >= self.succ_frame:
                # Blink is detected, so counting set to zero 
                # to start again when there is a new blink
                self.count_frame = 0
                blink = 1
            else:
            # When there is no blink 
                self.count_frame = 0
        
        # If the period of time of closed eyes is too long, the driver might be sleeping
        if self.count_frame > self.fps/2:
            blink = 2
            
        return blink, eye_closed

    def calculate_perclos(self, closed_eye: bool, length_of_frames: int):
        """Calculates the PERCLOS (percentage of eye closure) value 
        based on the number of frames the eye is closed

        Args:
            blink (bool): indicates whether the read frame 
            is closed (True) or open (False)
            length_of_frames (int): Defines the time span over which 
            the Perclos value is calculated

        Returns:
            perclos (float): Output of the PERCLOS value
        """
        #initialization
        perclos = 0
        number_of_frames = len(self.list_of_eye_closure)

        # Calculation when time span has been reached
        if number_of_frames == length_of_frames:
            
            # The oldest frame is removed and the new frame is added to the list
            self.list_of_eye_closure.append(closed_eye)
            self.list_of_eye_closure.pop(0)
            
            # Calculation of the Perclos value based on the values 
            # where eye is closed from the list
            frame_is_blink = self.list_of_eye_closure.count(True)
            perclos = frame_is_blink/number_of_frames

        # Collect frames until time span (in frames) has been reached
        elif number_of_frames < length_of_frames:

            self.list_of_eye_closure.append(closed_eye)

            # First of all, Perclos is a kind of loading value 
            # until a period of time has been reached
            perclos = (number_of_frames/length_of_frames)*100

        # Error message when list gets longer for some reason
        else:
            print("Fehler, Liste zu lang")

        return perclos
    
    def get_coord_points(self, landmark_list: list, eye_idxs: list, imgW: int, imgH: int):

        """Function for getting all six coordinate points of one eye

        Parameters:
            landmark_list (list): a list of all landmarks from the mediapipe face mesh
            Must be a list of 478 Landmarks

            eye_idxs (list): 6-entry large array of the corresponding landmarks 
            of the eye in the order: 
            [middle left, top right, top left, middle right, bottom right, bottom left]

        Returns:
            list: A List of the coordinate points
        """
        denormalize_coordinates = mp.solutions.drawing_utils._normalized_to_pixel_coordinates

        coords_points = []

        #Getting the (x,y) Coordinates of every Input-Landmark
        for i in eye_idxs:
            lm = landmark_list[i]
            coord = denormalize_coordinates(lm.x, lm.y, imgW, imgH)
            coords_points.append(coord)

        return coords_points

    def calculate_EAR(self, eye: list):
        """Function for calculating the EAR (Eye Aspect Ratio)

        Parameters:
            eye (list): 6-entry large array of the coordinate points (x, y)
            of the eye in the order: 
            [middle left, top right, top left, middle right, bottom right, bottom left]

            More informations about the EAR and the order of the coordinate points:
            https://www.mdpi.com/1866552 ; Dewi, C.; Chen, R.-C.; Chang, C.-W.; Wu, S.-H.; 
            Jiang, X.; Yu, H. Eye Aspect Ratio for Real-Time Drowsiness Detection 
            to Improve Driver Safety. Electronics 2022, 11, 3183.

        Returns:
            float: The calculated EAR value
        """
            
        # calculate the vertical distances
        vertical1 = dist.euclidean(eye[1], eye[5])
        vertical2 = dist.euclidean(eye[2], eye[4])
                
        # calculate the horizontal distance
        horizontal = dist.euclidean(eye[0], eye[3])
                
        # calculate the EAR
        EAR = (vertical1+vertical2)/(2*horizontal)
                    
        return EAR
    
    def get_list_of_ear(self, avg_ear: float, length: int):
        #initialization
        number_of_frames = len(self.list_of_EAR)
        blink_thresh = self.blink_thresh

        # Calculation when time span has been reached
        if number_of_frames == length:
            print("Fertig")
            self.list_of_EAR.append(avg_ear)
            pass
            
        # Collect EAR until time span (in frames) has been reached
        elif number_of_frames < length:

            self.list_of_EAR.append(avg_ear)

        # Pass when list gets longer for some reason
        else:
            pass

        return blink_thresh