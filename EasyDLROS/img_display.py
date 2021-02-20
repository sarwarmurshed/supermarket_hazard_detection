import os
import time
import cv2
import matplotlib.pyplot as plt


def show_images(directory):
 for filename in os.listdir(directory):
     path = directory + "/" + filename
     #im = Image.open(path)
     im = cv2.imread(path)
     #showPIL(im)
     cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
     cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
     cv2.imshow("window", im)

     cv2.waitKey(1000)
     cv2.destroyAllWindows()


show_images('/home/murshem/phd/git_research/rosres/runMXModel/new')
