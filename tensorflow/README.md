# TensorFlow benchmarks

Training neural networks using TensorFlow with GPU acceleration.

## Environment setup

### Apple Silicon Mac

Follow these instructions for installing on an Apple Silicon Mac:
https://developer.apple.com/metal/tensorflow-plugin/

Recommended to install in a new conda environment.
For example:

```bash
conda create -n benchmark-tensorflow python=3.9
conda activate benchmark-tensorflow
conda install -c apple tensorflow-deps
python -m pip install tensorflow-macos
python -m pip install tensorflow-metal
python -m pip install tensorflow_datasets
python -m pip install git+https://github.com/tensorflow/examples.git
```

### Other hardware

I have not tested on other setups.
Refer to TensorFlow documentation for installations and recommended environment setup.


## Running the benchmarks

Each subdirectory has a python script.
The python script will train the model and record the training time.
Run those scripts to obtain the benchmark numbers for your machine.