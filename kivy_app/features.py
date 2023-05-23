from scipy.spatial import distance as dist

# defining a function to calculate the EAR
def calculate_EAR(eye: list):
    """Function for calculating the EAR (Eye Aspect Ratio)

    Parameters:
        eye (list) : 6-entry large array of the coordinate points (x, y)
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
    EAR = (vertical1+vertical2)/2*horizontal
                
    return EAR
