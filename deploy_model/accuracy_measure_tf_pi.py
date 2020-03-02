#python3 accuracy_measure.py


#!/usr/bin/python

import os
import subprocess

import argparse

import numpy as np
import time
from PIL import Image
import tflite_runtime.interpreter as tf # TF2

def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]

def inference(model_file, label_file, image, input_mean = 127.5, input_std = 127.5):
  #print("Hello from label_image")

  interpreter = tf.Interpreter(model_path=model_file)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # check the type of the input tensor
  floating_model = input_details[0]['dtype'] == np.float32

  # NxHxWxC, H:1, W:2
  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]
  img = Image.open(image).resize((width, height))
  start_time = time.time()
  # add N dim
  input_data = np.expand_dims(img, axis=0)

  if floating_model:
    input_data = (np.float32(input_data) - input_mean) / input_std

  interpreter.set_tensor(input_details[0]['index'], input_data)

  interpreter.invoke()

  output_data = interpreter.get_tensor(output_details[0]['index'])
  itime = time.time() - start_time
  results = np.squeeze(output_data)

  top_k = results.argsort()[-2:][::-1]
  #print("what is top k: ", top_k)
  labels = load_labels(label_file)
  """for i in top_k:
    if floating_model:
      print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
    else:
      print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))
  """
  print('labels', labels[top_k[0]])
  return labels[top_k[0]], itime


path = os.getcwd()
script_path = '/usr/lib/python3/dist-packages/edgetpu/demo/'
#folders = []

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
   print ('\n\nImage name : ',key, 'Tag: ' , images[key], 'floor and the classification result is : ' )
   #start_time = time.time()
   res = inference('output_tflite_graph.tflite', 'labels.txt', key)
   #print("---%s seconds ---" % (time.time() - start_time))
   #print (res.decode("utf-8"))
   print ("return result", res[0], res[1])
   #print('key', key)
   total_time = total_time + res[1]
   if images[key] in res[0]:
      images[key] = 1
   else:
      images[key] = 0
      false_assum.update({key: 0})
#print("---%s seconds ---" % (time.time() - total_time))

print("total inference time", total_time/139)
print ("List of wrong assumtion : ", false_assum)
print ('########## Final result #########')
print ('Total right assumtion : ', sum(images.values()), '\nTotal worng assumtion : ', len(images)-sum(images.values()), '\nModel accuracy = ', round((sum(images.values())*100)/len(images), 3), '%')
