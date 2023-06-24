import joblib
from kivy.uix.screenmanager import Screen
import cv2
from kivy.core.audio import SoundLoader
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.logger import Logger
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist

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

        self.count_warning_frame_eyes_closed = 20

        self.count_warning_frame_classifier = 100

        # counter for capturing movement
        self.movement_counter = 0

        # Initialisation of list of frames for calculation of PERCLOS
        self.list_of_eye_closure = []

        # Initialisation of list of frames for calculation of blink threshold
        self.list_of_EAR = []

        self.list_of_blink_durations = []
        self.list_of_blink_frequency = []

        self.awake_ear_eyes_open = 0
        self.awake_perclos = 0.01
        self.awake_blink_duration = 0
        self.awake_avg_ear = 0

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
        self.initialize()
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
        self.ids.image_view.source = './assets/logo2_edit.png'

    def update(self, dt):
        
        if hasattr(self, 'capture') and hasattr(self, 'fps') and hasattr(self, 'face_mesh') and self.manager.current == 'detection':
            # Read a frame from the video capture
            if self.capture:
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
                            
                            # Testing, if the whole eye is detected
                            coord_points = coord_points_left + coord_points_right
                            if not any(item is None for item in coord_points):
                                
                                self.count_last +=1

                                #Calculating the Eye Aspect ratio for the left and right eye
                                EAR_left = self.calculate_EAR(coord_points_left)
                                
                                EAR_right = self.calculate_EAR(coord_points_right)

                                # Calculating the Average EAR for both eyes
                                avg_EAR = (EAR_right+EAR_left)/2

                                # Blink Detection Algorithm
                                blink, closed_eye, blink_duration = self.blink_detection(avg_EAR)

                                frame_length_perclos = 1000
                                frame_length_ear_list = 1000
                                num_of_blinks = 25



                                # PERCLOS Calculation based on frames
                                perclos = self.calculate_perclos(closed_eye, 
                                                                frame_length_perclos)
                                
                                # AVG EAR for eyes open
                                self.get_list_of_ear(avg_EAR, frame_length_ear_list)
                                avg_ear_eyes_open_at_test = self.avg_ear_eyes_open()
                                avg_ear_at_test = self.avg_ear()

                                avg_blink_duration = 1

                                # Counting the blinks
                                if blink == 1:
                                    self.blinks += 1
                                    avg_blink_duration = self.avg_blink_duration(blink_duration, num_of_blinks)

                                # Processing when the eye has been closed for too long
                                if blink == 2:
                                    if self.count_warning_frame_eyes_closed == 20:
                                        # Putting a text, that driver might be 
                                        # sleeping every 20 Frames
                                        cv2.putText(image, 'ALARM: Wake up!', (30, 30),
                                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)
                                        self.play_warning_sound()
                                        self.count_warning_frame_eyes_closed = 0
                                    else:
                                        self.count_warning_frame_eyes_closed +=1

                                # When the calibration is done
                                if self.cal_done:
                                    # Putting the PERCLOS value on Screen
                                    perclos_text = round(perclos, 2)
                                    string_perclos = "PERCLOS: " + str(perclos_text)
                                    cv2.putText(image, string_perclos, (30, 120),
                                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)

                                    # Generation of the feature vector
                                    feature_vector = self.feature_vector(perclos, 
                                                                        avg_blink_duration, 
                                                                        avg_ear_eyes_open_at_test, 
                                                                        avg_ear_at_test)

                                    # Prediction of the feature vector whether 
                                    # tired/half-tired/awake
                                    prediction = self.new_input(feature_vector)

                                    if prediction == 0:
                                        pass
                                        #TODO Visual apperance
                                    elif prediction == 1:
                                        pass
                                        #TODO Visual apperance
                                    else:
                                        if self.count_warning_frame_classifier == 100:
                                            self.count_warning_frame_classifier = 0
                                            self.play_warning_sound()
                                        self.count_warning_frame_classifier += 1
                                        #TODO Visual apperance

                                # If the Calibration is not done, continue the calibration
                                else:
                                    calibration = self.calibrate(
                                    frame_length_perclos, frame_length_ear_list, 
                                    perclos, avg_ear_eyes_open_at_test, num_of_blinks, 
                                    avg_blink_duration, avg_ear_at_test)
                                    
                                    # Putting a text for the calibration status
                                    calibration = round(calibration, 2)*100
                                    string_cal = "Calibration: " + str(calibration) + "%"
                                    cv2.putText(image, string_cal, (30, 120),
                                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)

                    else:
                        # if unable to detect landmarks for 100 frames,
                        # then give warning signs
                        self.movement_counter += 1
                        if self.movement_counter == 100:
                            self.play_warning_sound()
                            print("Landmarks nicht gefunden")
                            self.movement_counter = 0
                    
                    # Flip the image vertically for processing in kivy
                    buf1 = cv2.flip(image, 0)
                    buf = buf1.tostring()
                    image_texture = Texture.create(size=(image.shape[1], image.shape[0]), 
                    colorfmt='rgb')
                    image_texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
                    self.ids.image_view.texture = image_texture

    def play_warning_sound(self):
        tmp = True
        sound = SoundLoader.load('assets/mixkit-siren-tone-1649.wav')
        if sound and tmp:
            sound.play()
            tmp = False

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
            avg_ear_eyes_open (float): Mean of the EAR value 
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
            if len(list_of_eyes_open) > 0:
                avg_ear_eyes_open = sum(list_of_eyes_open) / len(list_of_eyes_open)
        
        return avg_ear_eyes_open
     
    def avg_ear(self):
        return np.mean(self.list_of_EAR)
    
    def avg_blink_duration(self, frame_blink_duration, length):
        print(frame_blink_duration)

        number_of_frames = len(self.list_of_blink_durations)

        avg_blink_duration = 1

        # Calculation when time span has been reached
        if number_of_frames == length:
            self.list_of_blink_durations.append(frame_blink_duration)
            self.list_of_blink_durations.pop(0)

            avg_blink_duration = np.mean(self.list_of_blink_durations)
            
        # Collect EAR until time span (in frames) has been reached
        elif number_of_frames < length:

            self.list_of_blink_durations.append(frame_blink_duration)
        
        return avg_blink_duration
    
    def calibrate(self, frame_length_perclos, frame_length_ear_list, perclos, ear_eyes_open, num_of_blinks, avg_blink_duration, avg_ear):
        """
        Calibrates the system based on the provided parameters.

        This method performs the calibration process for the system based on the frame 
        lengths for PERCLOS calculation and the list of average EAR values for eyes open. 

        Args:
            frame_length_perclos (int): The desired frame length for PERCLOS calculation.
            frame_length_ear_list (int): The desired frame length for the list of 
            average EAR values for eyes open.
            perclos (float): The current PERCLOS value.

            ear_eyes_open (float): The current average EAR value for eyes open.
        Returns:
            float: The calibration status as a decimal value indicating the 
            progress towards reaching the desired frame lengths.

        """      
        cal_perclos = False # Flag indicating if PERCLOS calibration is done
        cal_ear = False # Flag indicating if average EAR calibration is done
        cal_blinks = False
        calibrate_status = 0

        # Checking the length of the frame lists
        if frame_length_ear_list >= frame_length_perclos:
            calibrate_status = self.count_last/frame_length_ear_list
        else:
            calibrate_status = self.count_last/frame_length_perclos

        # Checking for the length of the PERCLOS List
        if self.count_last == frame_length_perclos and not cal_perclos:
            self.awake_perclos = perclos
            cal_perclos = True

        # Checking for the length of the EAR List
        if self.count_last == frame_length_ear_list and not cal_ear:
            self.awake_ear_eyes_open = ear_eyes_open
            self.awake_avg_ear = avg_ear
            cal_ear = True

        # Checking for the length of the Blink list
        if self.blinks == num_of_blinks and not cal_blinks:
            self.awake_blink_duration = avg_blink_duration
            cal_blinks = True

        # Checking if the frame length of PERCLOS and EAR is done
        if cal_ear and cal_perclos and cal_blinks:
            self.cal_done = True

        return calibrate_status
    
    def feature_vector(self, frame_perclos, frame_blink_duration, frame_avg_ear_eyes_open, frame_avg_ear):
        """
        Calculate the feature vector based on the difference between 
        the frame PERCLOS and the awake PERCLOS.

        Args:
            frame_perclos (float): PERCLOS value for the current frame.

        Returns:
            list: Feature vector containing the difference 
            between frame PERCLOS and awake PERCLOS.

        """
        ratio_avg_ear = frame_avg_ear/self.awake_avg_ear
        ratio_avg_ear_eyes_open = frame_avg_ear_eyes_open/self.awake_ear_eyes_open
        ratio_blink_duration = frame_blink_duration/self.awake_blink_duration
        ratio_perclos = frame_perclos/self.awake_perclos
        feature_vector = [ratio_perclos, ratio_blink_duration, ratio_avg_ear_eyes_open, ratio_avg_ear]

        print(feature_vector)
        #feature_vector = ratio_perclos
        return feature_vector
    
    def new_input(self, feature_vector):
        """
        Perform prediction using a loaded classifier based on the given feature vector.

        Args:
            feature_vector (list): Feature vector for the input.

        Returns:
            str: Prediction result.

        """

        # Load the Classifier
        loaded_classifier = joblib.load("best_classifier.pkl")

        # Predict the Class of the Feature Vector
        prediction = loaded_classifier.predict([feature_vector])

        return prediction
    

    def yawning():
        return 0