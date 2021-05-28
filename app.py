# This is the code for the app prototype.


# webcam capture code inspired by: https://github.com/kevinam99/capturing-images-from-webcam-using-opencv-python/blob/master/webcam-capture-v1.01.py

import cv2 

#--Prompt with instructions (including request for webcam access)





#--Take photo. (Access to webcam must be granted)

key = cv2. waitKey(1)
webcam = cv2.VideoCapture(0)
while True:
    try:
        check, frame = webcam.read()
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)

        #On 's' key pressed, photo is taken --> converted to greyscale --> saved as jpg

        if key == ord('s'):
            cv2.imwrite(filename='saved_img.jpg', img=frame)
            webcam.release()
            img_new = cv2.imread('saved_img.jpg', cv2.IMREAD_GRAYSCALE)
            img_new = cv2.imshow("Captured Image", img_new)
            cv2.waitKey(1650)
            cv2.destroyAllWindows()
            print("Processing image...")
            img_ = cv2.imread('saved_img.jpg', cv2.IMREAD_ANYCOLOR)
            print("Converting RGB image to grayscale...")
            gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
            print("Converted RGB image to grayscale...")
            cv2.imwrite(filename='save_grayscale.jpg', img=gray)
            print("Image saved!")
        
            break

        # On 'q' key pressed, window just closes without photograph being taken.


        elif key == ord('q'):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break
        
    #Exception: on key press (other than 's' or 'q') close window

    except(KeyboardInterrupt):
        print("Turning off camera.")
        webcam.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()
        break

#-- Run CNN prediciton on photo and let's see what happens +Show emotion e.g. "Are you *angry* ?" Y/N