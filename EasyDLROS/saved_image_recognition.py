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

from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

# Load the symbols for the networks
with open('synset.txt', 'r') as f:
    synsets = [l.rstrip() for l in f]

# Load the network parameters
sym, arg_params, aux_params = mx.model.load_checkpoint('mobilenetv1-1.0', 100)


# Load the network into an MXNet module and bind the corresponding parameters
mod = mx.mod.Module(symbol=sym, context=mx.cpu())
mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))])
mod.set_params(arg_params, aux_params)

def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model_dir',
      type=str,
      default='/tmp/imagenet',
      help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
  )
  parser.add_argument(
      '--image_file',
      type=str,
      default='',
      help='Absolute path to image file.'
  )
  parser.add_argument(
      '--num_top_predictions',
      type=int,
      default=5,
      help='Display this many predictions.'
  )
  global FLAGS
  FLAGS, unparsed = parser.parse_known_args()
  return unparsed

def predict(image_data, mod, synsets, N=5):
    
    tic = time.time()
    img = cv2.resize(image_data, (224, 224))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    #print("I am here")
    print ("pre-processed image in "+str(time.time()-tic))
    time.sleep(1)

    mod.forward(Batch([mx.nd.array(img)]))
    toc = time.time()
    prob = mod.get_outputs()[0].asnumpy()
    prob = np.squeeze(prob)
    rtime = time.time()-toc
    print ("forward pass in "+str(time.time()-toc))


    topN = []
    a = np.argsort(prob)[::-1]
    for i in a[0:N]:
        print('probability=%f, class=%s' %(prob[i], synsets[i]))
        topN.append((prob[i], synsets[i]))
    return topN, rtime

# Code to predict on a local file
def predict_from_local_file(image_data, N=1):
    return predict(image_data, mod, synsets, N)

class RosTensorFlow():
    def __init__(self):
        #self._session = tf.Session()

        self._cv_bridge = CvBridge()

        #self._sub = rospy.Subscriber('image', Image, self.callback, queue_size=1)
        self._pub = rospy.Publisher('result', String, queue_size=1)
        self.score_threshold = rospy.get_param('~score_threshold', 0.1)
        self.use_top_k = rospy.get_param('~use_top_k', 5)

        path = '/home/murshem/phd/git_research/mldledge/dataset/sdir/'
        files = os.listdir(path)
        #print (files)

        images = {}
        false_assum = {}

        for file in files:
                if 'hazard' in file:
                        images.update({file : 'hazard'})
                elif 'clean' in file:
                        images.update({file : 'clean'})

        #print (images)

        total_time = 0
        for key in images:
                print ('\n\nImage name : ',path+key, 'Tag: ' , images[key], 'floor and the classification result is : ' )
                cv_image = cv2.imread(path+key)
                if cv_image is None:
                        return None
                predictions, rtime = predict_from_local_file(cv_image, N=1)
                print ('MX result outer loop: ', predictions)
                self._pub.publish(predictions[0][1])

    '''

    def callback(self, image_msg):
        #cv_image = self._cv_bridge.imgmsg_to_cv2(image_msg, "rgb8")
        #cv2.imshow("Image window", cv_image)
        #cv2.waitKey(3)
        cv_image = cv2.imread('hazard.jpg')
        if cv_image is None:
                return None
        predictions, rtime = predict_from_local_file(cv_image, N=1)
        print ('MX result : ', predictions)
        self._pub.publish(predictions[0][1])
    '''

    def main(self):
        rospy.spin()

if __name__ == '__main__':
    setup_args()
    rospy.init_node('rostensorflow')
    tensor = RosTensorFlow()
    tensor.main()
