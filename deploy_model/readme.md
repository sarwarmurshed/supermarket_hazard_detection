### How to run DL models on Raspberry Pi
Method 1:
 1. Install [MXNet](https://mxnet.apache.org/get_started/?platform=devices&iot=raspberry-pi&) and necessary [libraries](https://mxnet.apache.org/api/python/docs/tutorials/deploy/inference/wine_detector.html) on Raspberry Pi. 
 2. Copy the model_name.params, model_name-symbol.json, which were generated after the model training, into the Raspberry Pi.
 3. Copy test-images to the Raspberry Pi.
 4. Run "accuracy_measure.py".

Method 2:
 1. Convert the ONNX model to tflite model using converter notebook
 2. Copy the model.tflite into the Raspberry Pi and Coral Dev Board.
 3. Copy test-images to the Raspberry Pi and Coral Dev Board.
 4. Run "accuracy_measure_tf_pi.py" on Raspberry Pi or "accuracy_measure_tf_cb.py" on Coral Dev Board to measure accuracy.

