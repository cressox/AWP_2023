from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.button import Button
from kivy.uix.camera import Camera
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.progressbar import ProgressBar
from kivy.uix.label import Label
from kivy.uix.switch import Switch
import cv2
import os
import dlib
import time
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
os.environ['KIVY_CAMERA'] = 'opencv'


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
        if self.progress_value < 100:
            self.progress_value += 1
            self.progress_bar.value = self.progress_value
        else:
            # After the ProgressBar reaches its maximum, go to DetectionScreen
            self.manager.current = 'detection'
            Clock.unschedule(self.progress_update_event)

class DetectionScreen2(Screen):
    def initialize(self):
        layout = BoxLayout(orientation='vertical')

        button_to_main = Button(text='Go to Main')
        button_to_main.bind(on_press=lambda x: self.set_screen('main'))
        layout.add_widget(button_to_main)

        self.image = Image()
        layout.add_widget(self.image)
        self.capture = cv2.VideoCapture(0)
        self.detector = dlib.get_frontal_face_detector()

        # Startet den Timer für die Bildaktualisierung
        Clock.schedule_interval(self.update, 0.05)
        self.add_widget(layout)

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            # Führe Gesichtserkennung aus
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)

            # Zeichne Rechtecke um erkannte Gesichter
            for rect in faces:
                x1 = rect.left()
                y1 = rect.top()
                x2 = rect.right()
                y2 = rect.bottom()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Konvertiere das Bild in ein Format, das von Kivy verwendet werden kann
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

            # Zeige das Bild an
            self.image.texture = image_texture

    def set_screen(self, screen_name):
        self.manager.current = screen_name

class DetectionScreen(Screen):
    def initialize(self):
        layout = BoxLayout(orientation='vertical')

        button_to_main = Button(text='Go to Main')
        button_to_main.bind(on_press=lambda x: self.set_screen('main'))
        layout.add_widget(button_to_main)

        self.image = Image()
        layout.add_widget(self.image)
        self.capture = cv2.VideoCapture(0)

        # Lade den dlib Detektor und den Shape Predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Path to the shape predictor model

        Clock.schedule_interval(self.update, 0.05)
        self.add_widget(layout)

    def update(self, dt):
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
        self.add_widget(MainScreen(name='main'))
        self.add_widget(DetectionScreen(name='detection'))
        self.add_widget(LoadingScreen(name='loading'))  # Add LoadingScreen
        self.add_widget(SettingsScreen(name='settings'))
        self.add_widget(HelpScreen(name='help'))

class MyApp(App):
    def build(self):
        return MyScreenManager()


if __name__ == '__main__':
    MyApp().run()
