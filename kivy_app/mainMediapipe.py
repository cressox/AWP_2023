from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.core.window import Window
from video_input_detection_mediapipe import DetectionScreen
import os
from kivy.graphics import Color, Rectangle
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.properties import ObjectProperty
from kivy.uix.screenmanager import SlideTransition
from kivy.config import Config

Config.set('input', 'mouse', 'mouse,disable_multitouch')


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
    """Bildschirm, der den Benutzern hilft, die Anwendung zu verstehen."""

class LoadingScreen(Screen):
    """Bildschirm, der nur temporär genutzt wird um laden von Instanzen zu überbrücken."""
    # def on_enter(self):
    #     time.sleep(2)
    #     MyScreenManager.set_screen(self, 'main')
    #     print("deteiction now")

class MyScreenManager(ScreenManager):
    """
    Klasse zur Verwaltung der verschiedenen Bildschirme der Anwendung.
    """
    mode = 'fast'  # Standardwert für den Modus
    def set_mode(self, mode):
        """
        Setzt den Modus der Anwendung.
        :param mode: Der Modus, zu dem gewechselt werden soll.
        :type mode: str
        :return: None
        """
        self.mode = mode

    def get_mode(self):
        """
        Gibt den aktuellen Modus der Anwendung zurück.
        :return: Der aktuelle Modus der Anwendung.
        :rtype: str
        """
        return self.mode

    def set_screen(self, screen_name):
        """
        Ändert den aktuell angezeigten Bildschirm.
        :param screen_name: Der Name des anzuzeigenden Bildschirms.
        :type screen_name: str
        :return: None
        """
        if screen_name == 'main':
            self.transition = SlideTransition(direction='right')
        else:
            self.transition = SlideTransition(direction='left')

        if screen_name == 'main' and self.current == 'detection':
            self.current = 'loading'
            Clock.schedule_once(lambda dt: self.load_detection_screen(), 5)
        else:
            self.current = screen_name

    def load_detection_screen(self):
        self.current = 'main'

class MyApp(App):
    """
    Hauptanwendungsklasse.
    """
    dark_mode = False
    color_scheme = ObjectProperty()
    font_scheme = ObjectProperty()
    show_main_button = BooleanProperty(True)

    def build(self):
        """
        Erstellt die Anwendung und legt die Fenstergröße fest.
        :return: Die ScreenManager-Instanz, die die Anwendung verwaltet.
        :rtype: MyScreenManager
        """
        Window.size = (360, 640)  # Adjust the window size for Samsung Galaxy S20 FE
        self.color_scheme = self.get_color_scheme()
        self.font_scheme = self.get_font_scheme()
        return MyScreenManager()

    def reSize(*args):
        Window.size = (360, 640) 
        return True

    def get_font_scheme(self):
        """
        Gibt das Schriftschema der Anwendung zurück.
        :return: Ein Wörterbuch, das das Schriftschema der Anwendung definiert.
        :rtype: dict
        """
        return {
            'font_name': 'assets/fonts/Roboto/Roboto-Medium.ttf',
            'font_size_back': 33,
            'font_size_settings': 40,
            'font_size_main': 40,
            'font_size_help': 23
        }

    def get_color_scheme(self):
        """
        Gibt das Farbschema der Anwendung zurück.
        :return: Ein Wörterbuch, das das Farbschema der Anwendung definiert.
        :rtype: dict
        """
        if self.dark_mode:
            return {
                'background': [0, 0, 0, 1],
                'button': [0.37, 0.37, 0.37, 1],
                'layout_element': [0.3, 0.3, 0.3, 0.3],
                'button_text': [1, 1, 1, 1],
                'button_hover': [0.1, 0.1, 0.1, 0.1],
                'img': 'assets/logo3_edit.png'
            }
        else:
            return {
                'background': [1, 1, 1, 1],
                'button': [0.1, 0.2, 0.3, 0.1],
                'layout_element': [0.5, 0.5, 0.5, 0.3],
                'button_text': [0, 0, 0, 1],
                'button_hover': [0.1, 0.1, 0.1, 0.1],
                'img': 'assets/logo2_edit.png'
            }

    def toggle_dark_mode(self):
        """
        Wechselt zwischen dem hellen und dunklen Modus der Anwendung.
        :return: None
        """
        self.dark_mode = not self.dark_mode
        self.color_scheme = self.get_color_scheme()

    Window.bind(on_resize = reSize)

if __name__ == '__main__':
    Builder.load_file('mainmediapipe.kv')
    MyApp().run()
