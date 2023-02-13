import cv2
import image_dehazer

if __name__ == "__main__":

    HazeImg = cv2.imread('Images/foggy_oaks.jpg')						# read input image -- (**must be a color image**)
    HazeCorrectedImg, haze_map = image_dehazer.remove_haze(HazeImg, showHazeTransmissionMap=False)		# Remove Haze

    cv2.imshow('haze_map', haze_map);						# display the original hazy image
    cv2.imshow('enhanced_image', HazeCorrectedImg);			# display the result
    cv2.waitKey(0)
    cv2.imwrite("outputImages/result.png", HazeCorrectedImg)