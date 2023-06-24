from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.core.window import Window
from face_detection_mediapipe import DetectionScreen
import os
from kivy.graphics import Color, Rectangle
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.properties import ObjectProperty
from kivy.uix.screenmanager import SlideTransition


os.environ['KIVY_CAMERA'] = 'opencv'

SCREENS = {}


class MainScreen(Screen):
    """Hauptbildschirm"""

class SettingsScreen(Screen):
    """Einstellungsbildschirm"""
    def __init__(self, **kwargs):
        super(SettingsScreen, self).__init__(**kwargs)
        self.app = App.get_running_app()
    global mode 
    mode = 'fast'  # Hinzugefügtes mode-Attribut

    def change_mode(self, mode):
        """
        Ändert den Modus basierend auf dem ausgewählten Modus.
        """
        if mode == 'detailed':
            # Detailierter Modus
            self.ids.detailed_mode_button.background_color = (0.5, 1, 0.5, 0.5)  # Light Green
            self.ids.fast_mode_button.background_color = self.app.color_scheme['button']  # Default button color


        else:
            # Schneller Modus
            self.ids.fast_mode_button.background_color = (0.5, 1, 0.5, 0.5)  # Light Green
            self.ids.detailed_mode_button.background_color = self.app.color_scheme['button']  # Default button color



class HelpScreen(Screen):
    """Hilfebildschirm"""


class MyScreenManager(ScreenManager):
    """Screen Manager"""

    mode = 'fast'  # Standardwert für den Modus

    def set_mode(self, mode):
        """
        Setzt den Modus.
        """
        self.mode = mode

    def get_mode(self):
        """
        Gibt den aktuellen Modus zurück.
        """
        return self.mode

    def set_screen(self, screen_name):
        """
        Wechselt zum angegebenen Screen.
        """
        if screen_name == 'main':
            self.transition = SlideTransition(direction='right')
        else:
            self.transition = SlideTransition(direction='left')
        self.current = screen_name


class MyApp(App):
    dark_mode = False 
    color_scheme = ObjectProperty()
    font_scheme = ObjectProperty()

    def build(self):
        """
        Erstellt die App und legt die Fenstergröße fest.
        """
        Window.size = (360, 640)  # Adjust the window size for Samsung Galaxy S20 FE
        self.color_scheme = self.get_color_scheme()
        self.font_scheme = self.get_font_scheme()
        return MyScreenManager()

    def get_font_scheme(self):
        return {
            'font_name': 'assets/fonts/Roboto/Roboto-Medium.ttf',
            'font_size_back': 25,
            'font_size_settings': 40,
            'font_size_main': 40,
            'font_size_help': 23
        }

    def get_color_scheme(self):
        if self.dark_mode:
            return {
                'background': [0, 0, 0, 1],
                'button': [0.37, 0.37, 0.37, 1],
                'layout_element': [0.3, 0.3, 0.3, 0.3],
                'button_text': [1, 1, 1, 1],
                'button_hover': [0.1, 0.1, 0.1, 0.1],
                'img':'assets/logo3_edit.png'
            }
        else:
            return {
                'background': [1, 1, 1, 1],
                'button': [0.1, 0.2, 0.3, 0.1],
                'layout_element': [0.5, 0.5, 0.5, 0.3],
                'button_text': [0, 0, 0, 1],
                'button_hover': [0.1, 0.1, 0.1, 0.1],
                'img':'assets/logo2_edit.png'
            }

    def toggle_dark_mode(self):
        self.dark_mode = not self.dark_mode
        self.color_scheme = self.get_color_scheme()

if __name__ == '__main__':
    Builder.load_file('mainmediapipe.kv')
    MyApp().run()
