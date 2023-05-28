from kivy.uix.screenmanager import Screen
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
import cv2
import threading
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
import mediapipe as mp
import numpy as np
import features as feat

#Class DetectionScreen is working in Kivy with Mediapipe Version 0.9.0

class DetectionScreen(Screen):
    def initialize(self):
        layout = BoxLayout(orientation='vertical')

        button_to_main = Button(text='Go to Main')
        button_to_main.bind(on_press=lambda x: self.set_screen('main'))
        layout.add_widget(button_to_main)

        self.image = Image()
        layout.add_widget(self.image)

        self.add_widget(layout)

        # Starte das Video in einem neuen Thread
        threading.Thread(target=self.initialize_resources).start()

        Clock.schedule_interval(self.update, 0.05)

    def initialize_resources(self):
        self.capture = cv2.VideoCapture(0)

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.denormalize_coordinates = self.mp_drawing._normalized_to_pixel_coordinates
        

        self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=False,         # Default=False
                    max_num_faces=1,                # Default=1
                    refine_landmarks=True,         # Default=False
                    min_detection_confidence=0.5,   # Default=0.5
                    min_tracking_confidence= 0.5,)

    def update(self, dt):
        if hasattr(self, 'capture'):
            ret, frame = self.capture.read()
            if ret:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                image = np.ascontiguousarray(image)
                imgH, imgW, _ = image.shape
                
                results = self.face_mesh.process(image)

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        self.mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                            )
                        # Landmark points corresponding to left eye
                        all_left_eye_idxs = list(self.mp_face_mesh.FACEMESH_LEFT_EYE)
                        # flatten and remove duplicate
                        all_left_eye_idxs = set(np.ravel(all_left_eye_idxs)) 
                        
                        # Landmark points corresponding to right eye
                        all_right_eye_idxs = list(self.mp_face_mesh.FACEMESH_RIGHT_EYE)
                        all_right_eye_idxs = set(np.ravel(all_right_eye_idxs))
                        
                        # Combined for plotting - Landmark points for both eye
                        all_idxs = all_left_eye_idxs.union(all_right_eye_idxs)
                        
                        # The chosen 12 points:   P1,  P2,  P3,  P4,  P5,  P6
                        chosen_left_eye_idxs  = [362, 385, 387, 263, 373, 380]
                        chosen_right_eye_idxs = [33,  160, 158, 133, 153, 144]
                        all_chosen_idxs = chosen_left_eye_idxs + chosen_right_eye_idxs
                        
                        for landmark_idx, landmark in enumerate(face_landmarks.landmark):
                            if landmark_idx in all_chosen_idxs:
                                pred_cord = self.denormalize_coordinates(landmark.x, 
                                                                    landmark.y, 
                                                                    imgW, imgH)
                                cv2.circle(image, 
                                        pred_cord,
                                        3,
                                        (255, 255, 255), 
                                        -1
                                        )

                        coords_points_left = []
                        for i in chosen_left_eye_idxs:
                            lm = face_landmarks.landmark[i]
                            coord = self.denormalize_coordinates(lm.x, lm.y, 
                                                            imgW, imgH)
                            coords_points_left.append(coord)
                        
                        ear = feat.calculate_EAR(coords_points_left)
                        print(ear)

                buf1 = cv2.flip(image, 0)
                buf = buf1.tostring()
                image_texture = Texture.create(size=(image.shape[1], image.shape[0]), 
                colorfmt='rgb')
                image_texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')

                self.image.texture = image_texture

    def set_screen(self, screen_name):
        self.manager.current = screen_name