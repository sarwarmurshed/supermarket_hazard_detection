### How to run DL models on Raspberry Pi
 1. Install [MXNet](https://mxnet.apache.org/get_started/?platform=devices&iot=raspberry-pi&) and necessary [libraries](https://mxnet.apache.org/api/python/docs/tutorials/deploy/inference/wine_detector.html) on Raspberry Pi. 
 2. Copy the model_name.params, model_name-symbol.json, which were generated after the model training, into the Raspberry Pi.
 3. Copy test-images to the Raspberry Pi.
 3. Run "accuracy_measure.py".
