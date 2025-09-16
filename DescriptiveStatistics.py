import matplotlib.pyplot as plt
import cv2

def find_value(img):
    image = cv2.imread(img)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    V = hsv_image[:,:,2]

    """
    plt.figure()
    plt.imshow(hsv,cmap='gray')
    plt.title("HSV image")
    plt.show()
    """

    return V


