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

class DetectionScreen(Screen):
    def initialize(self):
        threading.Thread(target=self.initialize_resources).start()
        Clock.schedule_interval(self.update, 0.05)

    def initialize_resources(self):
        self.capture = cv2.VideoCapture(0)

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("Predictors/shape_predictor_68_face_landmarks.dat")

        self.capture = None
        self.update_event = None
        self.draw_landmarks = True
        
        #Initialisierung der Werte für die Blinzeldetektion
        self.count_frame = 0
        self.blink_thresh = 0.18
        self.succ_frame = 1

        #Initialisierung der Liste der Frames für die PERCLOS Berechnung
        self.list_of_eye_closure = []

        #Initialisierung der Liste der Frames für die blink-Threshold Berechnung
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
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        if self.update_event is not None:
            Clock.unschedule(self.update_event)
            self.update_event = None
        print(self.blinks)
        print(self.awake_ear_eyes_open)
        print(self.awake_perclos)

    def update(self, dt):
        
        # Eye landmarks
        (L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

        if self.capture is not None:
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
                        
                    #Calculating the Eye Aspect ratio for the left and right eye
                    EAR_left = self.calculate_EAR(lefteye)
                        
                    EAR_right = self.calculate_EAR(righteye)

                    # Calculating the Average EAR for both eyes
                    avg_EAR = (EAR_right+EAR_left)/2

                    # Blink Detection Algorithm
                    blink, closed_eye = self.blink_detection(avg_EAR)

                    frame_length_perclos = 1000
                    frame_length_ear_list = 1000

                    #PERCLOS Calculation based on frames
                    perclos = self.calculate_perclos(closed_eye, 
                                                    frame_length_perclos)
                        
                    #AVG EAR for eyes open
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

        # Error message when list gets longer for some reason
        else:
            print("Fehler, Liste zu lang")

        return perclos

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
        
        # Elaborated features: difference awake status to current status + perclos value
        # Once ratio mean EAR value where eyes open and once ratio Perclos
        diff_ear_eyes_open = self.awake_ear_eyes_open-frame_ear_eyes_open
        diff_perclos = self.awake_perclos-frame_perclos
        perclos = frame_perclos

        return [diff_ear_eyes_open, diff_perclos, perclos]