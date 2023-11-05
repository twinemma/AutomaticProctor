# Setup Guide

## Requirements

[![Python 3.6](https://img.shields.io/badge/Python-3.6-3776AB)](https://www.python.org/downloads/release/python-360/)
[![TensorFlow 2.2](https://img.shields.io/badge/TensorFlow-2.2-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.2.0)
[![Protobuf Compiler >= 3.0](https://img.shields.io/badge/ProtoBuf%20Compiler-%3E3.0-brightgreen)](https://grpc.io/docs/protoc-installation/#install-using-a-package-manager)

## Installation

```bash
git clone git@github.com:twinemma/AutomaticProctor.git
cd AutomaticProctor/object_detection
```

Now we need to install the TensorFlow Object Detection API by cloning the TensorFlow Models repository.

```bash
git clone https://github.com/tensorflow/models.git
```

## Set up $PYTHONPATH

```bash
cd models/research
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
cd ..
export PYTHONPATH=$PYTHONPATH:`pwd`
```

This needs to be properly set to run some *.py code.

## Python Package Installation

```bash
cd models/research
# Compile protos.
protoc object_detection/protos/*.proto --python_out=.
# Install TensorFlow Object Detection API.
cp object_detection/packages/tf2/setup.py .
python -m pip install --use-feature=2020-resolver .
# Uninstall tensorflow 2.5.0 and install tensorflow 2.2.0
pip install tensorflow==2.2.0
pip install tf-models-official==2.2.0

# To check which version installed, use
pip show tensorflow
```

## Common Issues
If you run into issue of unknown option of "--use-feature", please upgrade your pip using
```bash
pip install --upgrade pip
```

If you run into issues in installing pycocotools in running "python -m pip install --use-feature=2020-resolver .", you may need to install "python3-dev". For example, on centOs, this can be installed through
```bash
sudo yum install python3-devel
```

If you run into issue of "protoc command not found", please install protobuf compiler 
```bash
PROTOC_ZIP=protoc-3.15.8-linux-x86_64.zip
curl -OL https://github.com/google/protobuf/releases/download/v3.15.8/$PROTOC_ZIP
sudo unzip -o $PROTOC_ZIP -d /usr/local bin/protoc
sudo unzip -o $PROTOC_ZIP -d /usr/local include/*
rm -f $PROTOC_ZIP
sudo chomod +rx /usr/local/bin/protoc
```

## Test Installation

```bash
# Test the installation.
python object_detection/builders/model_builder_tf2_test.py
```

If you run into SSL certification error, please try to run Applications/Python 3.6/Install Certificates.command from the terminal.

## COCO API Installation (Optional)

```bash
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools models/research/
```

## Steps to Train Custom Object Detector

### Partition DataSet into Train and Test DataSets
```bash
python partition_dataset.py -x -i AutomaticProctor/object_detection/workspace/training_demo/images -r 0.1
```
### Generate TFRecord for Train and Test DataSets
```bash
python generate_tfrecord.py -x AutomaticProctor/object_detection/workspace/training_demo/images/train -l AutomaticProctor/object_detection/workspace/training_demo/annotations/label_map.pbtxt -o AutomaticProctor/object_detection/workspace/training_demo/annotations/train.record

python generate_tfrecord.py -x AutomaticProctor/object_detection/workspace/training_demo/images/test -l AutomaticProctor/object_detection/workspace/training_demo/annotations/label_map.pbtxt -o AutomaticProctor/object_detection/workspace/training_demo/annotations/test.record
```

### Start Training Job
```bash
python model_main_tf2.py --model_dir=models/custom_ssd_mobilenet_v2_300x300 --pipeline_config_path=models/custom_ssd_mobilenet_v2_300x300/pipeline.config
```

### Start Evaluation Job
```bash
python model_main_tf2.py --model_dir=models/custom_ssd_mobilenet_v2_300x300 --pipeline_config_path=models/custom_ssd_mobilenet_v2_300x300/pipeline.config --checkpoint_dir=models/custom_ssd_mobilenet_v2_300x300
```
### Exported Trained Model
```bash
python exporter_main_v2.py --input_type image_tensor --pipeline_config_path models/custom_ssd_mobilenet_v2_300x300/pipeline.config --trained_checkpoint_dir models/custom_ssd_mobilenet_v2_300x300 --output_directory exported-models/custom_ssd_mobilenet_v2_300x300
```

