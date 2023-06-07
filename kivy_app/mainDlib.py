from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.core.window import Window

from face_detection_dlib import DetectionScreen

import cv2
import threading
import os
import dlib
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.lang import Builder

os.environ['KIVY_CAMERA'] = 'opencv'

SCREENS = {}


class MainScreen(Screen):
    pass


class LoadingScreen(Screen):
    def on_enter(self, **kwargs):
        super(LoadingScreen, self).on_enter(**kwargs)
        self.progress_value = 0
        self.progress_update_event = Clock.schedule_interval(self.update_progress_bar, 0.05)

    def update_progress_bar(self, dt):
        if self.ids.progress_bar.value == 100:
            self.manager.current = 'detection'
            Clock.unschedule(self.progress_update_event)
            self.ids.progress_bar.value = 0
        elif self.ids.progress_bar.value < 100:
            self.ids.progress_bar.value += 1


'''
class DetectionScreen(Screen):
    def initialize(self):
        threading.Thread(target=self.initialize_resources).start()
        Clock.schedule_interval(self.update, 0.05)

    def initialize_resources(self):
        self.capture = cv2.VideoCapture(0)

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("assets/shape_predictor_68_face_landmarks.dat")

    def update(self, dt):
        if hasattr(self, 'capture'):
            ret, frame = self.capture.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.detector(gray)
                for rect in faces:
                    x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    shape = self.predictor(gray, rect)
                    for i in range(shape.num_parts):
                        p = shape.part(i)
                        cv2.circle(frame, (p.x, p.y), 2, (0, 255, 0), -1)

                buf1 = cv2.flip(frame, 0)
                buf = buf1.tostring()
                image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
                image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

                self.ids.image_view.texture = image_texture
'''

class SettingsScreen(Screen):
    mode = 'fast'  # Hinzugefügtes mode-Attribut

    def toggle_mode(self, mode):
        if mode == 'fast':
            self.ids.detailed_mode_button.background_color = (0.2, 0.3, 0.4, 1)
            self.ids.fast_mode_button.background_color = (0.1, 0.2, 0.3, 1)
        else:
            self.ids.fast_mode_button.background_color = (0.2, 0.3, 0.4, 1)
            self.ids.detailed_mode_button.background_color = (0.1, 0.2, 0.3, 1)





class HelpScreen(Screen):
    pass


class MyScreenManager(ScreenManager):  
    mode = 'fast'  # Standardwert für den Modus

    def set_mode(self, mode):
        self.mode = mode

    def get_mode(self):
        return self.mode

    def load_detection_screen(self):
        self.get_screen('detection').initialize()
        self.current = 'loading'

    def set_screen(self, screen_name):
        self.current = screen_name


class MyApp(App):
    def build(self):
        Window.size = (360, 640)  # Adjust the window size for Samsung Galaxy S20 FE
        return MyScreenManager()


if __name__ == '__main__':
    Builder.load_file('maindlib.kv')
    MyApp().run()
