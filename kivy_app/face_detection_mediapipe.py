from kivy.uix.screenmanager import Screen
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
import cv2
import threading
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.logger import Logger
import mediapipe as mp
import numpy as np
import features as feat

# Class DetectionScreen is working in Kivy with Mediapipe Version 0.9.0

class DetectionScreen(Screen):
    def initialize(self):
        layout = BoxLayout(orientation='vertical')

        button_to_main = Button(text='Go to Main')
        button_to_main.bind(on_press=lambda x: self.set_screen('main'))
        layout.add_widget(button_to_main)

        self.image = Image()
        layout.add_widget(self.image)

        self.add_widget(layout)

        # Strating the video in a new Thread
        threading.Thread(target=self.initialize_resources).start()

        framerate = 50

        Clock.schedule_interval(self.update, 1/framerate)

        Logger.info("Base: Framerate is %s", framerate)


    def initialize_resources(self):

        self.capture = cv2.VideoCapture(0)

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
        
        self.count_frame = 0
        self.blink_thresh = 0.2
        self.succ_frame = 2

        Logger.info("Mediapipe: 478 Landmarks are detected")
        

    def update(self, dt):
        
        if hasattr(self, 'capture'):
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
                        # Landmarks of the eye and contours are drawn
                        self.mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                            )
                        
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
                        coord_points_left = feat.get_coord_points(
                            face_landmarks.landmark, left_eye_idxs, imgW, imgH)
                        
                        coord_points_right = feat.get_coord_points(
                            face_landmarks.landmark, right_eye_idxs, imgW, imgH)
                        
                        #Calculating the Eye Aspect ratio for the left and right eye
                        EAR_left = feat.calculate_EAR(coord_points_left)
                        
                        EAR_right = feat.calculate_EAR(coord_points_right)

                        # Calculating the Average EAR for both eyes
                        avg_EAR = (EAR_right+EAR_left)/2

                        # Blink Detection Algorithm

                        # Counting the frames when there is a blink
                        if avg_EAR < self.blink_thresh:
                            self.count_frame +=1

                        else:
                            # The blink is done, if the counting stops 
                            # and the EAR is bigger tan the blink threshold
                            if self.count_frame >= self.succ_frame:
                                # Blink is detected, so counting set to zero 
                                # to start again when there is a new blink
                                self.count_frame = 0
                                # Putting a text, that a blink is detected
                                cv2.putText(image, 'Blink Detected', (30, 30),
                                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)
                            else:
                                # When there is no blink 
                                self.count_frame = 0
                        
                buf1 = cv2.flip(image, 0)
                buf = buf1.tostring()
                image_texture = Texture.create(size=(image.shape[1], image.shape[0]), 
                colorfmt='rgb')
                image_texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')

                self.image.texture = image_texture

    def set_screen(self, screen_name):
        self.manager.current = screen_name