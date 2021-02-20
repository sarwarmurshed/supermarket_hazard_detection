import mxnet as mx
import numpy as np
import time
import cv2, os
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

def predict(filename, mod, synsets, N=5):
    tic = time.time()
    #img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    img = cv2.imread(filename)
    if img is None:
        return None
    img = cv2.resize(img, (224, 224))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    print ("pre-processed image in "+str(time.time()-tic))


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
    return topN,rtime


# Code to predict on a local file
def predict_from_local_file(filename, N=1):
    return predict(filename, mod, synsets, N)

#path = '/home/murshed/rosHazard/rosres/runMXModel/data/'
path = '/home/murshem/phd/git_research/rosres/runMXModel/data/'
files = os.listdir(path)
#print (files)

images = {}
false_assum = {}
total_images = 0

for file in files:
   if 'hazard' in file:
      images.update({file : 'hazard'})
      total_images += 1
   elif 'clean' in file:
      images.update({file : 'clean'})
      total_images += 1

#print (images)

total_time = 0
for key in images:
   print ('\n\nImage name : ',path+key, 'Tag: ' , images[key], 'floor and the classification result is : ' )
   res, rtime = predict_from_local_file(path+key, N=1)
   print (res[0][1])
   print(rtime)
   total_time = total_time + rtime
   if images[key] in res[0]:
      images[key] = 1
   else:
      images[key] = 0
      false_assum.update({key: 0})
#print("---%s seconds ---" % (time.time() - total_time))

#print('total_images = ', total_images)
print("avg process time: ", total_time/total_images)
print ("List of wrong assumtion : ", false_assum)
print ('########## Final result #########')
print ('Total right assumtion : ', sum(images.values()), '\nTotal worng assumtion : ', len(images)-sum(images.values()), '\nModel accuracy = ', round((sum(images.values())*100)/len(images), 3), '%')

