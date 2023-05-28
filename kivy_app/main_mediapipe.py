from kivy.config import Config
from kivy.core.audio import SoundLoader
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
import cv2
import os
import mediapipe as mp

Config.set('input', 'wm_pen', '0')

os.environ['KIVY_CAMERA'] = 'opencv'

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)

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
        self.manager.current = 'detection'
        
    def set_screen(self, screen_name):
        self.manager.current = screen_name

class DetectionScreen(Screen):
    def __init__(self, **kwargs):
        super(DetectionScreen, self).__init__(**kwargs)
        layout = BoxLayout(orientation='vertical')

        button_to_main = Button(text='Go to Main', size_hint=(None, None), size=(100, 50), pos_hint={'x': 0, 'y': 1})
        button_to_main.bind(on_press=lambda x: self.set_screen('main'))
        layout.add_widget(button_to_main)

        self.image = Image()
        layout.add_widget(self.image)

        self.add_widget(layout)

        self.capture = None
        self.update_event = None
        self.draw_landmarks = True

    def on_enter(self):
        self.start_camera()

    def on_leave(self):
        self.stop_camera()

    def start_camera(self):
        self.capture = cv2.VideoCapture(0)
        self.update_event = Clock.schedule_interval(self.update, 0.05)

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
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = face_mesh.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if self.draw_landmarks and results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            image, face_landmarks, connections=mp_face_mesh.FACEMESH_TESSELATION)

                buf1 = cv2.flip(image, 0)
                buf = buf1.tobytes()
                image_texture = Texture.create(
                    size=(image.shape[1], image.shape[0]), colorfmt='bgr')
                image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

                self.image.texture = image_texture

    def play_warning_sound(self):
        sound = SoundLoader.load('warning.ogg')
        if sound:
            sound.play()

    def set_screen(self, screen_name):
        self.manager.current = screen_name


class SettingsScreen(Screen):
    def __init__(self, **kwargs):
        super(SettingsScreen, self).__init__(**kwargs)
        layout = BoxLayout(orientation='vertical')

        button_to_main = Button(text='Go to Main', size_hint=(None, None), size=(100, 50), pos_hint={'x': 0, 'y': 1})
        button_to_main.bind(on_press=lambda x: self.set_screen('main'))
        layout.add_widget(button_to_main)

        self.buffer_label = Label(text='BUFFER_SIZE: 128')
        layout.add_widget(self.buffer_label)

        buffer_slider = Slider(min=32, max=256, value=128)
        buffer_slider.bind(value=self.on_buffer_size_changed)
        layout.add_widget(buffer_slider)

        self.add_widget(layout)

        self.detection_screen = None

    def set_screen(self, screen_name):
        self.manager.current = screen_name

    def on_buffer_size_changed(self, instance, value):
        BUFFER_SIZE = int(value)
        self.buffer_label.text = f'BUFFER_SIZE: {BUFFER_SIZE}'
        # Hier kannst du die BUFFER_SIZE verwenden, wie du möchtest

class HelpScreen(Screen):
    def __init__(self, **kwargs):
        super(HelpScreen, self).__init__(**kwargs)
        layout = BoxLayout(orientation='vertical')

        button_to_main = Button(text='Go to Main', size_hint=(None, None), size=(100, 50), pos_hint={'x': 0, 'y': 1})
        button_to_main.bind(on_press=lambda x: self.set_screen('main'))
        layout.add_widget(button_to_main)

        self.add_widget(layout)

    def on_enter(self):
        DetectionScreen().play_warning_sound()

    def set_screen(self, screen_name):
        self.manager.current = screen_name

class MyScreenManager(ScreenManager):
    def __init__(self, **kwargs):
        super(MyScreenManager, self).__init__(**kwargs)
        SCREENS['main'] = MainScreen(name='main')
        SCREENS['detection'] = DetectionScreen(name='detection')
        SCREENS['settings'] = SettingsScreen(name='settings')
        SCREENS['help'] = HelpScreen(name='help')

        self.add_widget(SCREENS['main'])
        self.add_widget(SCREENS['detection'])
        self.add_widget(SCREENS['settings'])
        self.add_widget(SCREENS['help'])

class MyApp(App):
    def build(self):
        return MyScreenManager()

if __name__ == '__main__':
    MyApp().run()
