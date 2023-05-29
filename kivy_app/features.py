from scipy.spatial import distance as dist
import mediapipe as mp

def calculate_EAR(eye: list):
    """Function for calculating the EAR (Eye Aspect Ratio)

    Parameters:
        eye (list): 6-entry large array of the coordinate points (x, y)
        of the eye in the order: 
        [middle left, top right, top left, middle right, bottom right, bottom left]

        More informations about the EAR and the order of the coordinate points:
        https://www.mdpi.com/1866552 ; Dewi, C.; Chen, R.-C.; Chang, C.-W.; Wu, S.-H.; 
        Jiang, X.; Yu, H. Eye Aspect Ratio for Real-Time Drowsiness Detection 
        to Improve Driver Safety. Electronics 2022, 11, 3183.

    Returns:
        float: The calculated EAR value
    """
        
    # calculate the vertical distances
    vertical1 = dist.euclidean(eye[1], eye[5])
    vertical2 = dist.euclidean(eye[2], eye[4])
            
    # calculate the horizontal distance
    horizontal = dist.euclidean(eye[0], eye[3])
            
    # calculate the EAR
    EAR = (vertical1+vertical2)/(2*horizontal)
                
    return EAR

def get_coord_points(landmark_list: list, eye_idxs: list, imgW: int, imgH: int):

    """Function for getting all six coordinate points of one eye

    Parameters:
        landmark_list (list): a list of all landmarks from the mediapipe face mesh
        Must be a list of 478 Landmarks

        eye_idxs (list): 6-entry large array of the corresponding landmarks 
        of the eye in the order: 
        [middle left, top right, top left, middle right, bottom right, bottom left]

    Returns:
        list: A List of the coordinate points
    """
    denormalize_coordinates = mp.solutions.drawing_utils._normalized_to_pixel_coordinates

    coords_points = []

    #Getting the (x,y) Coordinates of every Input-Landmark
    for i in eye_idxs:
        lm = landmark_list[i]
        coord = denormalize_coordinates(lm.x, lm.y, imgW, imgH)
        coords_points.append(coord)

    return coords_points

