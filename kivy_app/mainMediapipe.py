from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.core.window import Window
from video_input_detection_mediapipe import DetectionScreen
import os
from kivy.clock import Clock
from kivy.lang import Builder


os.environ['KIVY_CAMERA'] = 'opencv'

SCREENS = {}


class MainScreen(Screen):
    """Hauptbildschirm"""


class LoadingScreen(Screen):
    """Ladescreen"""

    def on_enter(self, **kwargs):
        """
        Wird aufgerufen, wenn der Screen aktiviert wird.
        Startet den Fortschrittsbalken.
        """
        super(LoadingScreen, self).on_enter(**kwargs)
        self.progress_value = 0
        self.progress_update_event = Clock.schedule_interval(
            self.update_progress_bar, 0.05
        )

    def update_progress_bar(self, dt):
        """
        Aktualisiert den Fortschrittsbalken.
        Wenn der Balken 100 erreicht hat, wechselt der Screen zum Erkennungsscreen.
        """
        if self.ids.progress_bar.value == 100:
            self.manager.current = 'detection'
            Clock.unschedule(self.progress_update_event)
            self.ids.progress_bar.value = 0
        elif self.ids.progress_bar.value < 100:
            self.ids.progress_bar.value += 1


class SettingsScreen(Screen):
    """Einstellungsbildschirm"""

    mode = 'fast'  # Hinzugefügtes mode-Attribut

    def toggle_mode(self, mode):
        """
        Ändert den Modus basierend auf dem ausgewählten Modus.
        """
        if mode == 'fast':
            self.ids.detailed_mode_button.background_color = (0.2, 0.3, 0.4, 1)
            self.ids.fast_mode_button.background_color = (0.1, 0.2, 0.3, 1)
        else:
            self.ids.fast_mode_button.background_color = (0.2, 0.3, 0.4, 1)
            self.ids.detailed_mode_button.background_color = (0.1, 0.2, 0.3, 1)


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

    def load_detection_screen(self):
        """
        Lädt den Erkennungsscreen.
        """
        self.get_screen('detection').initialize()
        self.current = 'loading'

    def set_screen(self, screen_name):
        """
        Wechselt zum angegebenen Screen.
        """
        self.current = screen_name


class MyApp(App):
    def build(self):
        """
        Erstellt die App und legt die Fenstergröße fest.
        """
        Window.size = (360, 640)  # Adjust the window size for Samsung Galaxy S20 FE
        return MyScreenManager()


if __name__ == '__main__':
    Builder.load_file('mainmediapipe.kv')
    MyApp().run()
