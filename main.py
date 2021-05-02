# Import the required module
import cv2 as cv
from random import randrange

# Grabbing the dataset
car_training_data_set = cv.CascadeClassifier('car_dataSet.xml')

def Detect_car_images():
    """This function is responsible for converting an image into gray scale"""

    img = cv.imread('Images/car1.jpg')
    
    cv.imshow("Car1",img)
    
    gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    

    def draw_rect():
        """This function can draw rectangles across face coordinates"""
        car_face_coordinates = car_training_data_set.detectMultiScale(gray)

        for (x,y,w,h) in car_face_coordinates:
            rect_car = cv.rectangle(img,(x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)),2)

        cv.imshow("Dog Detected",rect_car)
    draw_rect()


# main function
if __name__ == "__main__":

    Detect_car_images()
    
    cv.waitKey(0)
    cv.destroyAllWindows()
    print("Code Completed 🔥")

