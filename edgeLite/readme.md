### Description
 This repository contains necessary code and instructions to build, train and fine-tune EdgeLite model on grocery hazard detection.

### Prepare IMAGENET Data
1. You are advised to use CUDA-compatible GPUs to prepare data and train the model.
2. Download [IMAGENET](http://www.image-net.org/) from official Website
3. Split data into train and test directory using [image_process.py](https://github.com/sarwarmurshed/supermarket_hazard_detection/blob/master/edgeLite/image_process.py)
4. Convert images to rec formate using [imagenet.py](https://github.com/sarwarmurshed/supermarket_hazard_detection/blob/master/edgeLite/imagenet.py)
### Train EdgeLite model
1. Follow [edgeLite.ipynb](https://github.com/sarwarmurshed/supermarket_hazard_detection/blob/master/edgeLite/edgeLite.ipynb). This notebook has code to train and evaluate the EdgeLite model.
