from kivy.uix.screenmanager import Screen
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
import cv2
import threading
import dlib
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from imutils import face_utils
import features as feat

class DetectionScreen(Screen):
    def initialize(self):
        layout = BoxLayout(orientation='vertical')

        button_to_main = Button(text='Go to Main')
        button_to_main.bind(on_press=lambda x: self.set_screen('main'))
        layout.add_widget(button_to_main)

        self.image = Image()
        layout.add_widget(self.image)

        self.add_widget(layout)

        #Starting the video in a new Thread
        threading.Thread(target=self.initialize_resources).start()

        Clock.schedule_interval(self.update, 0.05)

    def initialize_resources(self):
        self.capture = cv2.VideoCapture(0)
  
        # loading the dlib face detector
        self.detector = dlib.get_frontal_face_detector()
        # Path to the shape predictor model
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    def update(self, dt):            
        
        # Variables
        blink_thresh = 0.45
        count_frame = 0
        
        # Eye landmarks
        (L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

        if hasattr(self, 'capture'):
            ret, frame = self.capture.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.detector(gray)
                
                for rect in faces:
                    # Draw a rectangle around the detected face
                    x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()  # noqa: E501
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    # Get the landmarks for the face and draw them
                    shape = self.predictor(gray, rect)
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
        
                    # Calculate the EAR
                    left_EAR = feat.calculate_EAR(lefteye)
                    right_EAR = feat.calculate_EAR(righteye)
        
                    # Avg of left and right eye EAR
                    avg = (left_EAR+right_EAR)/2
                    if avg < blink_thresh:
                        count_frame += 1  # incrementing the frame count
                    else:
                        cv2.putText(frame, 'Blink Detected', (30, 30),
                                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)
                        count_frame = 0
                # Convert the image to a format that Kivy can use
                buf1 = cv2.flip(frame, 0)
                buf = buf1.tostring()
                image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), 
                                               colorfmt='bgr')
                image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

                self.image.texture = image_texture

    def set_screen(self, screen_name):
        self.manager.current = screen_name
