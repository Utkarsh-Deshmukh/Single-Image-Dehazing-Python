"""
Created on Thursday June 11 16:38:42 2020
@author: Utkarsh Deshmukh
"""
import cv2
import numpy as np

from Airlight import Airlight
from BoundCon import BoundCon
from CalTransmission import CalTransmission
from removeHaze import removeHaze

if __name__ == '__main__':
    HazeImg = cv2.imread('../Images/foggy_bench.jpg')

    # Resize image
    '''
    Channels = cv2.split(HazeImg)
    rows, cols = Channels[0].shape
    HazeImg = cv2.resize(HazeImg, (int(0.4 * cols), int(0.4 * rows)))
    '''

    # Estimate Airlight
    windowSze = 15
    AirlightMethod = 'fast'
    A = Airlight(HazeImg, AirlightMethod, windowSze)

    # Calculate Boundary Constraints
    windowSze = 3
    C0 = 20         # Default value = 20 (as recommended in the paper)
    C1 = 300        # Default value = 300 (as recommended in the paper)
    Transmission = BoundCon(HazeImg, A, C0, C1, windowSze)                  #   Computing the Transmission using equation (7) in the paper

    # Refine estimate of transmission
    regularize_lambda = 1       # Default value = 1 (as recommended in the paper) --> Regularization parameter, the more this  value, the closer to the original patch wise transmission
    sigma = 0.5
    Transmission = CalTransmission(HazeImg, Transmission, regularize_lambda, sigma)     # Using contextual information

    # Perform DeHazing
    HazeCorrectedImg = removeHaze(HazeImg, Transmission, A, 0.85)

    cv2.imshow('Original', HazeImg)
    cv2.imshow('Result', HazeCorrectedImg)
    cv2.waitKey(0)

    #cv2.imwrite('outputImages/result.jpg', HazeCorrectedImg)
