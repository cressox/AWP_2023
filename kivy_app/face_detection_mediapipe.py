from kivy.uix.screenmanager import Screen
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
import cv2
from kivy.core.audio import SoundLoader
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.logger import Logger
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import threading

class DetectionScreen(Screen):
    def initialize(self):
        Clock.schedule_once(self.initialize_resources)

    def initialize_resources(self,n):
        self.image = Image()

        Clock.schedule_interval(self.update, 0.02)

        self.fps = 0
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
        
        # Initialisation of values for blink detection
        self.count_frame = 0
        self.blink_thresh = 0.16
        self.succ_frame = 1

        self.count_warning_frame = 20

        # Initialisation of list of frames for calculation of PERCLOS
        self.list_of_eye_closure = []

        # Initialisation of list of frames for calculation of blink threshold
        self.list_of_EAR = []

        self.awake_ear_eyes_open = 0
        self.awake_perclos = 0.01

        self.count_last = -1
        self.cal_done = False

        self.blinks = 0

        # Select the 6 landmarks per eye 
        # for the calculation of the eye aspect ratio
        # The chosen 12 points:   P1,  P2,  P3,  P4,  P5,  P6
        self.left_eye_idxs  =    [362, 385, 387, 263, 373, 380]
        self.right_eye_idxs =    [33,  160, 158, 133, 153, 144]

        self.eye_idxs = self.left_eye_idxs + self.right_eye_idxs

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
        print(self.blinks)
        self.blinks = 0
        print(self.awake_ear_eyes_open)
        print(self.awake_perclos)

    def update(self, dt):
        if hasattr(self, 'capture') and hasattr(self, 'fps') and hasattr(self, 'face_mesh') and self.manager.current == 'detection':
            ret, frame = self.capture.read()
            if ret:
                # Changing to RGB so that mediapipe can process the frame
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                image = np.ascontiguousarray(image)
                imgH, imgW, _ = image.shape
                
                # Generation of the face mesh
                results = self.face_mesh.process(image)

                self.count_last +=1

                # Processing of the landmarks
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:

                        # Drawing the 6 landmarks per eye
                        for landmark_idx, landmark in enumerate(
                            face_landmarks.landmark):
                            if landmark_idx in self.eye_idxs:
                                pred_cord = self.denormalize_coordinates(
                                    landmark.x, landmark.y, imgW, imgH)
                                cv2.circle(image, pred_cord,3,(255, 255, 255), -1)

                        # Getting the coordinate points for left and right eye
                        coord_points_left = self.get_coord_points(
                            face_landmarks.landmark, self.left_eye_idxs, imgW, imgH)
                        
                        coord_points_right = self.get_coord_points(
                            face_landmarks.landmark, self.right_eye_idxs, imgW, imgH)
                        
                        #Calculating the Eye Aspect ratio for the left and right eye
                        EAR_left = self.calculate_EAR(coord_points_left)
                        
                        EAR_right = self.calculate_EAR(coord_points_right)

                        # Calculating the Average EAR for both eyes
                        avg_EAR = (EAR_right+EAR_left)/2

                        # Blink Detection Algorithm
                        blink, closed_eye, blink_duration = self.blink_detection(avg_EAR)

                        frame_length_perclos = 1000
                        frame_length_ear_list = 1000

                        # PERCLOS Calculation based on frames
                        perclos = self.calculate_perclos(closed_eye, 
                                                         frame_length_perclos)
                        
                        # AVG EAR for eyes open
                        self.get_list_of_ear(avg_EAR, frame_length_ear_list)
                        avg_ear_eyes_open_at_test = self.avg_ear_eyes_open()
                        
                        calibration = self.calibrate(
                            frame_length_perclos, frame_length_ear_list, 
                            perclos, avg_ear_eyes_open_at_test)

                        if self.cal_done:
                            perclos = round(perclos, 2)
                            string_perclos = "PERCLOS: " + str(perclos)
                            cv2.putText(image, string_perclos, (30, 120),
                            cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)

                            feature_vector = self.feature_vector(
                            avg_ear_eyes_open_at_test, perclos)

                        else:
                            calibration = round(calibration, 2)*100
                            string_cal = "Calibration: " + str(calibration) + "%"
                            cv2.putText(image, string_cal, (30, 120),
                            cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)

                        if blink == 1:
                            # Putting a text, that a blink is detected
                            cv2.putText(image, 'Blink Detected', (30, 30),
                            cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)
                            self.blinks +=1
                            print(self.blinks)
                            print(blink_duration) # prints nothing, why is it skipped? 

                        if blink == 2:
                            if self.count_warning_frame == 20:
                                # Putting a text, that driver might be sleeping
                                cv2.putText(image, 'ALARM: Wake up!', (30, 30),
                                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)
                                self.play_warning_sound()
                                self.count_warning_frame = 0
                            else:
                                self.count_warning_frame +=1
                        
                buf1 = cv2.flip(image, 0)
                buf = buf1.tostring()
                image_texture = Texture.create(size=(image.shape[1], image.shape[0]), 
                colorfmt='rgb')
                image_texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')

                self.ids.image_view.texture = image_texture

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
            the eye is closed or not
            blink_duration: duration of the blink in frames
        """
        blink = 0
        blink_duration = 0
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
                # save number of frames for blink duration
                blink_duration = self.count_frame
                self.count_frame = 0
                blink = 1
            else:
            # When there is no blink 
                self.count_frame = 0
        
        # If the period of time of closed eyes is too long, the driver might be sleeping
        if self.count_frame > self.fps/2:
            blink = 2
            
        return blink, eye_closed, blink_duration

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

        # Error message when list gets longer for some reason
        else:
            print("Fehler, Liste zu lang")

        return perclos
    
    def get_coord_points(self, landmark_list: list, eye_idxs: list, imgW: int, imgH: int):  # noqa: E501
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
        denormalize_coordinates = mp.solutions.drawing_utils._normalized_to_pixel_coordinates  # noqa: E501

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
            https://www.mdpi.com/1866552 ; 
            Dewi, C.; Chen, R.-C.; Chang, C.-W.; Wu, S.-H.; 
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
        """For a given length of frames a list ist build 
        with the EAR for the last frames

        Args:
            avg_ear (float): EAR Value for a given Frame
            length (int): The length of the list eg. of the collected frames
        """
        # Initialization
        number_of_frames = len(self.list_of_EAR)

        # Calculation when time span has been reached
        if number_of_frames == length:
            self.list_of_EAR.append(avg_ear)
            self.list_of_EAR.pop(0)
            
        # Collect EAR until time span (in frames) has been reached
        elif number_of_frames < length:

            self.list_of_EAR.append(avg_ear)

        # Pass when list gets longer for some reason
        else:
            pass
    
    def avg_ear_eyes_open(self):
        """Returns us over the length of the specified lists 
        at values where the eye is open is the mean

        Returns
            avg_ear_eyes_open(float): Mean of the EAR value 
            over a specified time when the eyes are open
        """
        # Initialise
        list_of_eyes_open = []
        avg_ear_eyes_open = -1
        # If the lengths of the two lists EAR and Augen zu yes/no are not the same, 
        # it cannot be guaranteed that the entries belong together
        if len(self.list_of_eye_closure) != len(self.list_of_EAR):
            print("Längen stimmen nicht überein")
        else:
            # Iterate over the entire length of the lists, 
            # entry in list_of_eyes_open if eye is open
            list_of_eyes_open = [self.list_of_EAR[i] for i in 
                                range(len(self.list_of_eye_closure)) 
                                if not self.list_of_eye_closure[i]]
            # Calculating the average
            avg_ear_eyes_open = sum(list_of_eyes_open) / len(list_of_eyes_open)
        
        return avg_ear_eyes_open
    
    def calibrate(self, frame_length_perclos, frame_length_ear_list, perclos, ear_eyes_open):  # noqa: E501
        
        # Storage of the first data of the awake status
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
        
        # elaborated features: difference awake status to current status + perclos value
        # Once ratio mean EAR value where eyes open and once ratio Perclos
        diff_ear_eyes_open = frame_ear_eyes_open/self.awake_ear_eyes_open
        diff_perclos = frame_perclos/self.awake_perclos

        return [diff_ear_eyes_open, diff_perclos]
    
    #TODO
    def movement():
        return 0
    
    def yawning():
        return 0