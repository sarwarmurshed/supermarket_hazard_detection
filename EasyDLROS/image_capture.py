import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import mxnet as mx
import argparse
import time
import os

#np.set_printoptions(threshold=np.inf)


base_path = "/home/murshem/phd/git_research/rosres/runMXModel/test"

class CaptureImage():
    def __init__(self):
        #self._session = tf.Session()
        self._cv_bridge = CvBridge()
        
        self.index = 0
        self.counter = 0
        self._sub = rospy.Subscriber('image', Image, self.callback, queue_size=1)
        #self._pub = rospy.Publisher('result', String, queue_size=1)
        #self.score_threshold = rospy.get_param('~score_threshold', 0.1)
        #self.use_top_k = rospy.get_param('~use_top_k', 5)

    def callback(self, image_msg):
        #rospy.sleep(1)
        cv_image = self._cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
        #cv2.imshow("Image window", cv_image)

        #start saving image
        
        print ("Saved to: ", base_path+str(self.index)+".jpg")
        if (self.counter == 10):
            cv2.imwrite(os.path.join(base_path, "frame{:06d}.jpg".format(self.index)), cv_image)
            self.counter = 1
        else:
            print("Not save: ", self.index, ".jpg")
        self.index += 1
        self.counter += 1
        #print(cv_image)  
        cv_image = 0
        image_msg = 0
        cv2.destroyAllWindows()
        
    def main(self):
        rospy.spin()



if __name__ == '__main__':
    #setup_args()
    rospy.init_node('captureImage')
    tensor = CaptureImage()
    tensor.main()
