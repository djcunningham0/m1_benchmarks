# TensorFlow benchmarks

Training neural networks using TensorFlow with GPU acceleration.

## Environment setup

### Apple Silicon Mac

Follow these instructions for installing on an Apple Silicon Mac:
https://developer.apple.com/metal/tensorflow-plugin/

Recommended to install in a new conda environment.
For example:

```bash
# create conda environment and install tensorflow
conda create -n benchmark-tensorflow python=3.9  # python 3.8 or 3.9 supported
conda activate benchmark-tensorflow
conda install -c apple tensorflow-deps
python -m pip install tensorflow-macos
python -m pip install tensorflow-metal

# install other requirements
python -m pip install tensorflow_datasets
python -m pip install git+https://github.com/tensorflow/examples.git
```

### Intel Mac

Follow the same instructions as before:
https://developer.apple.com/metal/tensorflow-plugin/

but do the steps for "x86: AMD". (*Note:* you must use python version 3.8 on an Intel Mac.)

Specifically:
```bash
# create virtual environment and install tensorflow
python -m venv venv_tensorflow_benchmarks  # must be python version 3.8!!!
. venv_tensorflow_benchmarks/bin/activate
python -m pip install --upgrade pip
python -m pip install tensorflow-macos
python -m pip install tensorflow-metal
python -m pip install keras==2.6.\*  # if keras==2.7.\* is installed it will cause errors

# install other requirements
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
