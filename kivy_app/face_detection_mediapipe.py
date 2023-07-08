import joblib
from kivy.uix.screenmanager import Screen
import cv2
from kivy.app import App
import time
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
from kivy.properties import BooleanProperty
from kivy.graphics import Color
from kivy.properties import ListProperty

class DetectionScreen(Screen):
    color = ListProperty([0.3, 0.3, 0.3, 0.3])  # Weiß

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

        # Define the Face Mesh object
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,         # Default=False
            max_num_faces=1,                 # Default=1
            refine_landmarks=True,           # Default=False
            min_detection_confidence=0.5,    # Default=0.5
            min_tracking_confidence=0.5,     # Default=0.5
        )

        # Initialize values for blink detection
        self.count_frame = 0
        self.blink_thresh = 0.16
        self.succ_frame = 1

        # Initialize the Counting length for repetition of warning sound
        self.seconds_warning_eyes_closed = 1.0
        self.seconds_warning_classification = 10.0
        self.count_warning_frame_classifier = 0

        # Counter for capturing movement
        self.movement_counter = 0

        # Lists to store eye closure, EAR, blink durations, and predictions
        self.list_of_eye_closure = []
        self.list_of_EAR = []
        self.list_of_blink_durations = []
        self.list_of_predictions = []

        # Initialize values for awake state
        self.awake_ear_eyes_open = 0
        self.awake_perclos = 0.01
        self.awake_blink_duration = 0
        self.awake_avg_ear = 0

        # Flags to control calculations
        self.cal_done = False
        
        # Initialisation fpr time calculations
        self.fps = 28.0
        self.calibration_start_time = time.time()
        self.elapsed_time = 0.0

        # Predicition, Initialize with 0 = Awake
        self.median_prediction = 0

        # Indices of landmarks for left and right eyes
        self.left_eye_idxs = [362, 385, 387, 263, 373, 380]
        self.right_eye_idxs = [33, 160, 158, 133, 153, 144]

        self.eye_idxs = self.left_eye_idxs + self.right_eye_idxs

        Logger.info("Mediapipe: 478 Landmarks are detected")

    def on_enter(self):
        """
        Event handler called when entering the screen.

        It starts the camera by calling the `start_camera` function.

        Parameters:
            None

        Returns:
            None
        """
        self.initialize()
        self.start_camera()
        self.initialize_resources(0)

    def on_leave(self):
        """
        Event handler called when leaving the screen.

        It stops the camera by calling the `stop_camera` function.

        Parameters:
            None

        Returns:
            None
        """
        self.stop_camera()

    def start_camera(self):
        """
        Starts the camera capture.

        It initializes the video capture using `cv2.VideoCapture`.
        The frames per second (fps) is obtained from the capture.
        An update event is scheduled using `Clock.schedule_interval` to call the `update` function.

        Parameters:
            None

        Returns:
            None
        """
        self.capture = cv2.VideoCapture(0)
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.update_event = Clock.schedule_interval(self.update, 1/self.fps)
        print(self.fps)

    def stop_camera(self):
        """
        Stops the camera capture.

        It releases the video capture using `release` and cancels the update event using `Clock.unschedule`.

        Parameters:
            None

        Returns:
            None
        """
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        if self.update_event is not None:
            Clock.unschedule(self.update_event)
            self.update_event = None
        self.blinks = 0
        self.ids.image_view.source = './assets/logo2_edit.png'

    def update(self, dt):
        """
        Update function called at regular intervals to process frames from the camera.

        It performs the following tasks:
        - Reads a frame from the video capture.
        - Converts the frame to RGB color format for processing with Mediapipe.
        - Generates the face mesh using Mediapipe.
        - Processes the landmarks and performs blink detection.
        - Calculates PERCLOS, average EAR, and other metrics.
        - Detects the Status: Awake/Questionable/Tired
        - Displays visual indicators and text based on the detection results.

        Parameters:
            dt (float): The time interval since the last update in seconds.

        Returns:
            None
        """        
        if hasattr(self, 'capture') and hasattr(self, 'fps') and hasattr(self, 'face_mesh') and self.manager.current == 'detection':
            # Read a frame from the video capture
            if self.capture:
                ret, frame = self.capture.read()
                self.elapsed_time = time.time() - self.calibration_start_time
                if ret:
                    # Changing to RGB so that mediapipe can process the frame
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    image = np.ascontiguousarray(image)
                    imgH, imgW, _ = image.shape
                    
                    # Generation of the face mesh
                    results = self.face_mesh.process(image)

                    # Processing of the landmarks
                    if results.multi_face_landmarks:
                        self.color = [0.3, 0.3, 0.3, 0.3]
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
                                
                                # Calculating the Eye Aspect ratio for the left and right eye
                                EAR_left = self.calculate_EAR(coord_points_left)
                                
                                EAR_right = self.calculate_EAR(coord_points_right)

                                # Calculating the Average EAR for both eyes
                                avg_EAR = (EAR_right+EAR_left)/2

                                # Blink Detection Algorithm
                                blink, closed_eye, blink_duration = self.blink_detection(avg_EAR)

                                # Defining the length for the calibration
                                time_length = 60 # 1 Minute Duration
    
                                # PERCLOS Calculation based on seconds
                                perclos = self.calculate_perclos(closed_eye, 
                                                                time_length)
                                
                                # AVG EAR for eyes open
                                self.get_list_of_ear(avg_EAR, time_length)
                                avg_ear_eyes_open_at_test = self.avg_ear_eyes_open()
                                avg_ear_at_test = self.avg_ear()
                                avg_blink_duration = self.avg_blink_duration(blink_duration, time_length, blink)
                                
                                # Warning Sound, if the eye is closed too long
                                if closed_eye:
                                    self.closed_frames += 1
                                    if self.closed_frames/self.fps >= self.seconds_warning_eyes_closed:
                                        cv2.putText(image, 'ALARM: Wake up!', (30, 30),
                                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)
                                        self.play_warning_sound()
                                        self.closed_frames = 0
                                else:
                                    self.closed_frames = 0

                                # When the calibration is done
                                if self.cal_done:
                                    # Generation of the feature vector
                                    feature_vector = self.feature_vector(perclos, 
                                                                        avg_blink_duration, 
                                                                        avg_ear_eyes_open_at_test, 
                                                                        avg_ear_at_test)

                                    # Prediction of the feature vector whether 
                                    # tired/half-tired/awake
                                    single_prediction = self.new_input(feature_vector)
                                    length = self.fps*30

                                    # Median Prediction
                                    pred = self.prediction_median(single_prediction, length)
                                    
                                    pred = int(pred)

                                    # Visual and auditive apperance depending on prediction
                                    if pred == 0:
                                        # Putting the Prediction value on Screen
                                        string_perclos = "Wach"
                                        cv2.putText(image, string_perclos, (30, 120),
                                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)
                                        #TODO Visual apperance
                                    elif pred == 1:
                                        # Putting the Prediction value on Screen
                                        string_perclos = "Fraglich"
                                        cv2.putText(image, string_perclos, (30, 120),
                                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)
                                        #TODO Visual apperance
                                    else:
                                        # Putting the Prediction value on Screen
                                        string_perclos = "Müde"
                                        cv2.putText(image, string_perclos, (30, 120),
                                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)

                                        # Visual Apperance and Sound, if the person is tired
                                        if self.count_warning_frame_classifier/self.fps == self.seconds_warning_classification:
                                            self.count_warning_frame_classifier = 0
                                            self.play_warning_sound()
                                        self.count_warning_frame_classifier += 1
                                        #TODO Visual apperance

                                # If the Calibration is not done, continue the calibration
                                else:
                                    calibration = self.calibrate(
                                    perclos, avg_ear_eyes_open_at_test,
                                    avg_blink_duration, avg_ear_at_test, time_length)
                                    
                                    # Putting a text for the calibration status
                                    calibration = round(calibration, 2)*100
                                    string_cal = "Calibration: " + str(calibration) + "%"
                                    cv2.putText(image, string_cal, (30, 120),
                                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)

                        else:
                            # If unable to detect landmarks for 3 seconds,
                            # then give warning signs
                            self.movement_counter += 1
                            if self.movement_counter/self.fps == 180:
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
        """
        Plays a warning sound.

        This method loads and plays a warning sound from the 'warning.ogg' file. 
        """
        tmp = True
        if self.color == [0.3, 0.3, 0.3, 0.3]:
            self.color = [1, 0, 0, 1] # Rot 
        else:
            self.color = [0.3, 0.3, 0.3, 0.3]
        sound = SoundLoader.load('assets/mixkit-siren-tone-1649.wav')

    def set_screen(self, screen_name):
        """
        Set the current screen to the specified screen name.

        Parameters:
        - screen_name: The name of the target screen to switch to.

        Returns:
        None
        """
        self.manager.current = screen_name 

    def blink_detection(self, avg_EAR: float):
        """
        Calculates the blink behavior based on the EAR value. 
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
            
        return blink, eye_closed, blink_duration

    def calculate_perclos(self, closed_eye: bool, length: int):
        """
        Calculates the PERCLOS (percentage of eye closure) value 
        based on the number of frames the eye is closed

        Args:
            blink (bool): indicates whether the read frame 
            is closed (True) or open (False)
            length (int): Defines the time span over which 
            the Perclos value is calculated

        Returns:
            perclos (float): Output of the PERCLOS value
        """

        # Initialization
        perclos = 0
        seconds_elapsed = int(self.elapsed_time)

        # Calculation when time span has been reached
        if seconds_elapsed >= length:
            
            # The oldest frame is removed and the new frame is added to the list
            self.list_of_eye_closure.append(closed_eye)
            self.list_of_eye_closure.pop(0)
            
            # Calculation of the Perclos value based on the values 
            # where eye is closed from the list
            frame_is_blink = self.list_of_eye_closure.count(True)
            perclos = frame_is_blink/(seconds_elapsed*self.fps)

        # Collect frames until time span (in frames) has been reached
        elif seconds_elapsed < length:

            self.list_of_eye_closure.append(closed_eye)

        # Error message when list gets longer for some reason
        else:
            print("Fehler, Liste zu lang")

        return perclos
    
    def get_coord_points(self, landmark_list: list, eye_idxs: list, imgW: int, imgH: int):
        """
        Function for getting all six coordinate points of one eye

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
        """
        Function for calculating the EAR (Eye Aspect Ratio)

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
        """
        For a given length of frames a list ist build 
        with the EAR for the last frames

        Args:avg_blink_duration
            avg_ear (float): EAR Value for a given Frame
            length (int): The length of the list eg. of the collected frames
        """
        # Initialization
        seconds_elapsed = self.elapsed_time

        # Calculation when time span has been reached
        if seconds_elapsed == length:
            self.list_of_EAR.append(avg_ear)
            self.list_of_EAR.pop(0)
            
        # Collect EAR until time span (in seconds) has been reached
        elif seconds_elapsed < length:

            self.list_of_EAR.append(avg_ear)

        # Pass when list gets longer for some reason
        else:
            pass
    
    def avg_ear_eyes_open(self):
        """
        Returns us over the length of the specified lists 
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
        """
        Calculate and return the average Eye Aspect Ratio (EAR) from the list of EAR values.

        Returns:
        - The average EAR value as a float.

        """
        return np.mean(self.list_of_EAR)
    
    def avg_blink_duration(self, frame_blink_duration, length, blink):
        """
        Calculate and return the average blink duration based on the given frame blink duration.

        Args:
        - frame_blink_duration (float): The duration of the current blink frame.
        - length (int): The desired length of the time span in seconds for calculating the average.

        Returns:
        - The average blink duration as a float.
        """

        # Calculating the blink duration
        if np.any(self.list_of_blink_durations):
            avg_blink_duration = np.mean(self.list_of_blink_durations)
        else:
            avg_blink_duration = 1.0

        if blink == 1:
            # Calculation when time span has been reached
            if self.elapsed_time == length:
                self.list_of_blink_durations.append((frame_blink_duration/self.fps)*1000)
                self.list_of_blink_durations.pop(0)

                avg_blink_duration = np.mean(self.list_of_blink_durations)
                
            # Collect EAR until time span (in frames) has been reached
            elif self.elapsed_time < length:

                self.list_of_blink_durations.append((frame_blink_duration/self.fps)*1000)
                avg_blink_duration = np.mean(self.list_of_blink_durations)
        
        return avg_blink_duration
    
    def calibrate(self, perclos, ear_eyes_open, avg_blink_duration, avg_ear, time_length):
        """
        Calibrates the system based on the provided parameters.

        This method performs the calibration process for the system. 

        Args:
            perclos (float): The current PERCLOS value.
            ear_eyes_open (float): The current average EAR value for eyes open.
            avg_blink_duration (float): The average blink duration in ms.
            avg_ear (float): The current average EAR value.
            time_length (float): The time length for the calibration in seconds
        Returns:
            float: The calibration status as a decimal value indicating the 
            progress towards reaching the desired time length.

        """      
        calibrate_status = 0

        # Checking the elapsed time
        if time_length >= self.elapsed_time:
            calibrate_status = self.elapsed_time/time_length

        # Defining the awake status values, when calibration time length is reached
        if time_length <= self.elapsed_time:
            self.awake_perclos = perclos
            self.awake_ear_eyes_open = ear_eyes_open
            self.awake_avg_ear = avg_ear
            self.awake_blink_duration = avg_blink_duration
            self.cal_done = True
            calibrate_status = 100.0

        return calibrate_status
    
    def feature_vector(self, frame_perclos, frame_blink_duration, frame_avg_ear_eyes_open, frame_avg_ear):
        """
        Calculate the feature vector based on the ratios of PERCLOS, Average EAR for eyes open, Average EAR and Blink duration

        Args:
            frame_perclos (float): PERCLOS value for the current frame.
            frame_blink_duration (float): Blink duration for the current frame.
            frame_avg_ear_eyes_open (float): Average EAR for eyes open for the current frame.
            frame_avg_ear (float): Average EAR for the current frame.

        Returns:
            list: Feature vector containing the ratios between the frame and awake values of PERCLOS, blink duration, average EAR for eyes open, and average EAR.

        """
        
        # Calculate the Ratios
        ratio_avg_ear = frame_avg_ear/self.awake_avg_ear
        ratio_avg_ear_eyes_open = frame_avg_ear_eyes_open/self.awake_ear_eyes_open
        ratio_blink_duration = frame_blink_duration/self.awake_blink_duration
        ratio_perclos = frame_perclos/self.awake_perclos

        return [ratio_perclos, ratio_blink_duration, ratio_avg_ear_eyes_open, ratio_avg_ear]
    
    def new_input(self, feature_vector):
        """
        Make a prediction based on the given feature vector using the pre-trained classifier.

        Args:
            feature_vector (list): Feature vector containing the ratios of PERCLOS, blink duration, average EAR for eyes open, and average EAR.

        Returns:
            int: Prediction indicating the state of drowsiness (0: awake, 1: half-tired, 2: tired).

        """
        classifier_all  = joblib.load("best_classifier.pkl")
        
        # Getting the predicition based on the loaded classifier
        if feature_vector[0] < 1.0:
            prediction = 0
        else:
            prediction = classifier_all.predict([feature_vector])
            prediction = prediction[0]
            print(prediction)

        return prediction
    
    def prediction_median(self, prediction, length):
        """
        Calculate the median prediction based on a given prediction value.

        Args:
            prediction (int): Prediction value indicating the state of drowsiness (0: awake, 1: half-tired, 2: tired).
            length (int): Length of the time span (in seconds) for calculating the median.

        Returns:
            float: Median prediction value.

        """
        number_of_frames = len(self.list_of_predictions)
        self.median_prediction = 0

        # Calculation when time span has been reached
        if number_of_frames == length:
            self.list_of_predictions.append(prediction)
            self.list_of_predictions.pop(0)
            self.median_prediction = np.median(self.list_of_predictions)
            
        # Collect predictions until time span has been reached
        elif number_of_frames < length:
            self.list_of_predictions.append(prediction)
        
        print(self.list_of_predictions)

        return self.median_prediction  
