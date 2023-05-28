import os
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.progressbar import ProgressBar
from kivy.uix.label import Label
from kivy.uix.switch import Switch
from kivy.clock import Clock
from face_detection_dlib import DetectionScreen

os.environ["KIVY_CAMERA"] = "opencv"

SCREENS = {}


class MainScreen(Screen):
    """
    This is the main screen of the app, it has buttons to navigate to other screens.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation="vertical")

        button_to_detection = Button(text="Go to Detection")
        button_to_detection.bind(on_press=self.load_detection_screen)
        layout.add_widget(button_to_detection)

        button_to_settings = Button(text="Go to Settings")
        button_to_settings.bind(on_press=lambda x: self.set_screen("settings"))
        layout.add_widget(button_to_settings)

        button_to_help = Button(text="Go to Help")
        button_to_help.bind(on_press=lambda x: self.set_screen("help"))
        layout.add_widget(button_to_help)

        self.add_widget(layout)

    def load_detection_screen(self, instance):
        """
        Initializes the DetectionScreen and then switch to the LoadingScreen.
        """
        self.manager.get_screen("detection").initialize()
        self.manager.current = "loading"

    def set_screen(self, screen_name):
        """
        Switches the current screen to the given screen name.
        """
        self.manager.current = screen_name


class LoadingScreen(Screen):
    """
    This is the loading screen of the app, it shows a progress bar.
    """

    def on_enter(self, **kwargs):
        super().on_enter(**kwargs)
        layout = BoxLayout(orientation="vertical")

        # Create the ProgressBar and add it to the layout
        self.progress_bar = ProgressBar(max=100)
        layout.add_widget(self.progress_bar)

        self.progress_value = 0
        self.progress_update_event = Clock.schedule_interval(
            self.update_progress_bar, 0.02
        )

        self.add_widget(layout)

    def update_progress_bar(self, dt):
        """
        Update the progress bar, switch to the detection screen when ready.
        """
        if hasattr(SCREENS["detection"], "capture"):
            self.manager.current = "detection"
            Clock.unschedule(self.progress_update_event)
        elif self.progress_value < 100:
            self.progress_value += 1
            self.progress_bar.value = self.progress_value
        else:
            self.progress_value = 0
            self.progress_bar.value = self.progress_value

class SettingsScreen(Screen):
    """
    This is the settings screen of the app, it has switches to change app settings.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation="vertical")

        button_to_main = Button(text="Go to Main")
        button_to_main.bind(on_press=lambda x: self.set_screen("main"))
        layout.add_widget(button_to_main)

        layout.add_widget(Label(text="Setting 1"))
        layout.add_widget(Switch())

        layout.add_widget(Label(text="Setting 2"))
        layout.add_widget(Switch())

        self.add_widget(layout)

    def set_screen(self, screen_name):
        """
        Switches the current screen to the given screen name.
        """
        self.manager.current = screen_name


class HelpScreen(Screen):
    """
    This is the help screen of the app, it provides information about 
    how to use the app.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation="vertical")

        button_to_main = Button(text="Go to Main")
        button_to_main.bind(on_press=lambda x: self.set_screen("main"))
        layout.add_widget(button_to_main)

        self.add_widget(layout)

    def set_screen(self, screen_name):
        """
        Switches the current screen to the given screen name.
        """
        self.manager.current = screen_name


class MyScreenManager(ScreenManager):
    """
    This is the main screen manager of the app, it manages all the screens.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        SCREENS["main"] = MainScreen(name="main")
        SCREENS["detection"] = DetectionScreen(name="detection")
        SCREENS["loading"] = LoadingScreen(name="loading")  # Add LoadingScreen
        SCREENS["settings"] = SettingsScreen(name="settings")
        SCREENS["help"] = HelpScreen(name="help")

        self.add_widget(SCREENS["main"])
        self.add_widget(SCREENS["detection"])
        self.add_widget(SCREENS["loading"])
        self.add_widget(SCREENS["settings"])
        self.add_widget(SCREENS["help"])


class MyApp(App):
    """
    This is the main app class, it builds the app with the screen manager.
    """

    def build(self):
        return MyScreenManager()


if __name__ == "__main__":
    MyApp().run()