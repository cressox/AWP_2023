from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.button import Button
from kivy.uix.camera import Camera
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.progressbar import ProgressBar
from kivy.uix.label import Label
from kivy.uix.switch import Switch
import cv2
import threading
import os
import dlib
import time
import math
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from scipy.spatial import distance as dist
import numpy as np
from imutils import face_utils

os.environ['KIVY_CAMERA'] = 'opencv'

SCREENS = {}

class MainScreen(Screen):
    def __init__(self, **kwargs):
        super(MainScreen, self).__init__(**kwargs)
        layout = BoxLayout(orientation='vertical')

        button_to_detection = Button(text='Go to Detection')
        button_to_detection.bind(on_press=self.load_detection_screen)
        layout.add_widget(button_to_detection)

        button_to_settings = Button(text='Go to Settings')
        button_to_settings.bind(on_press=lambda x: self.set_screen('settings'))
        layout.add_widget(button_to_settings)

        button_to_help = Button(text='Go to Help')
        button_to_help.bind(on_press=lambda x: self.set_screen('help'))
        layout.add_widget(button_to_help)

        self.add_widget(layout)

    def load_detection_screen(self, instance):
        # Initialize the DetectionScreen and then switch to the LoadingScreen
        self.manager.get_screen('detection').initialize()
        self.manager.current = 'loading'

    def set_screen(self, screen_name):
        self.manager.current = screen_name

class LoadingScreen(Screen):
    def on_enter(self, **kwargs):
        super(LoadingScreen, self).on_enter(**kwargs)
        layout = BoxLayout(orientation='vertical')

        # Create the ProgressBar and add it to the layout
        self.progress_bar = ProgressBar(max=100)
        layout.add_widget(self.progress_bar)

        self.progress_value = 0
        self.progress_update_event = Clock.schedule_interval(self.update_progress_bar, 0.05)

        self.add_widget(layout)

    def update_progress_bar(self, dt):
        if hasattr(SCREENS['detection'], 'capture'):
            self.manager.current = 'detection'
            Clock.unschedule(self.progress_update_event)
        elif self.progress_value < 100:
            self.progress_value += 1
            self.progress_bar.value = self.progress_value
        else:
            self.progress_value = 0
            self.progress_bar.value = self.progress_value

class DetectionScreen(Screen):
    def initialize(self):
        layout = BoxLayout(orientation='vertical')

        button_to_main = Button(text='Go to Main')
        button_to_main.bind(on_press=lambda x: self.set_screen('main'))
        layout.add_widget(button_to_main)

        self.image = Image()
        layout.add_widget(self.image)

        self.add_widget(layout)

        # Starte das Laden der Modelle und das Starten der Videoaufnahme in einem separaten Thread
        threading.Thread(target=self.initialize_resources).start()

        Clock.schedule_interval(self.update, 0.05)

    def initialize_resources(self):
        self.capture = cv2.VideoCapture(0)
  
        # Lade den dlib Detektor und den Shape Predictor
        self.detector = dlib.get_frontal_face_detector()
        # Path to the shape predictor model
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  

    def update(self, dt):

        # defining a function to calculate the EAR
        def calculate_EAR(eye):
        
            # calculate the vertical distances
            y1 = dist.euclidean(eye[1], eye[5])
            y2 = dist.euclidean(eye[2], eye[4])
        
            # calculate the horizontal distance
            x1 = dist.euclidean(eye[0], eye[3])
        
            # calculate the EAR
            EAR = (y1+y2) / x1
            return EAR
        
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
                    # Zeichne ein Rechteck um das erkannte Gesicht
                    x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    # Erhalte die Landmarks für das Gesicht und zeichne sie
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
                    left_EAR = calculate_EAR(lefteye)
                    right_EAR = calculate_EAR(righteye)
        
                    # Avg of left and right eye EAR
                    avg = (left_EAR+right_EAR)/2
                    # Avg of left and right eye EAR
                    if avg < blink_thresh:
                        count_frame += 1  # incrementing the frame count
                    else:
                        cv2.putText(frame, 'Blink Detected', (30, 30),
                                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)
                        count_frame = 0
        
                # Konvertiere das Bild in ein Format, das von Kivy verwendet werden kann
                buf1 = cv2.flip(frame, 0)
                buf = buf1.tostring()
                image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
                image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

                self.image.texture = image_texture

    def set_screen(self, screen_name):
        self.manager.current = screen_name

class SettingsScreen(Screen):
    def __init__(self, **kwargs):
        super(SettingsScreen, self).__init__(**kwargs)
        layout = BoxLayout(orientation='vertical')

        button_to_main = Button(text='Go to Main')
        button_to_main.bind(on_press=lambda x: self.set_screen('main'))
        layout.add_widget(button_to_main)

        layout.add_widget(Label(text='Einstellung 1'))
        layout.add_widget(Switch())

        layout.add_widget(Label(text='Einstellung 2'))
        layout.add_widget(Switch())

        self.add_widget(layout)

    def set_screen(self, screen_name):
        self.manager.current = screen_name

class HelpScreen(Screen):
    def __init__(self, **kwargs):
        super(HelpScreen, self).__init__(**kwargs)
        layout = BoxLayout(orientation='vertical')

        button_to_main = Button(text='Go to Main')
        button_to_main.bind(on_press=lambda x: self.set_screen('main'))
        layout.add_widget(button_to_main)

        self.add_widget(layout)

    def set_screen(self, screen_name):
        self.manager.current = screen_name

class MyScreenManager(ScreenManager):
    def __init__(self, **kwargs):
        super(MyScreenManager, self).__init__(**kwargs)
        SCREENS['main']=MainScreen(name='main')
        SCREENS['detection']=DetectionScreen(name='detection')
        SCREENS['loading']=LoadingScreen(name='loading')
        SCREENS['settings']=SettingsScreen(name='settings')
        SCREENS['help']=HelpScreen(name='help')

        self.add_widget(SCREENS['main'])
        self.add_widget(SCREENS['detection'])
        self.add_widget(SCREENS['loading'])  # Add LoadingScreen
        self.add_widget(SCREENS['settings'])
        self.add_widget(SCREENS['help'])


class MyApp(App):
    def build(self):
        return MyScreenManager()


if __name__ == '__main__':
    MyApp().run()
