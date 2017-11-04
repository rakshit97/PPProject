# LBP with CUDA

## Installation Guide

Link for CUDA installation guide on Linux: http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

To use CUDA on your system, you will need the following installed:
1. CUDA-capable GPU
2. A supported version of Linux with a gcc compiler and toolchain
3. NVIDIA CUDA Toolkit (available at http://developer.nvidia.com/cuda-downloads)

The CUDA development environment relies on tight integration with the host development environment, including the host compiler and C runtime libraries, and is therefore only supported on distribution versions that have been qualified for this CUDA Toolkit release.

Verify CUDA capable GPU
$ lspci | grep -i nvidia

Verify supported version of LINUX
$ uname -m && cat /etc/\*release

Verify GCC on system
$ gcc --version

Run following commands on UBUNTU to install CUDA
$ sudo dpkg -i cuda-repo-<distro>_<version>_<architecture>.deb
$ sudo apt-key add /var/cuda-repo-<version>/7fa2af80.pub
$ sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/<distro>/<architecture>/7fa2af80.pub
$ sudo apt-get update
$ sudo apt-get install cuda

### PyCUDA Installation

$ tar xfz pycuda-VERSION.tar.gz

$ cd pycuda-VERSION
$ su -c "python distribute_setup.py" # this will install distribute
$ su -c "easy_install numpy" # this will install numpy using distribute

$ cd pycuda-VERSION # if you're not there already
$ python configure.py --cuda-root=/where/ever/you/installed/cuda
$ su -c "make install"

$ cd pycuda-VERSION/test
$ python test_driver.py

### OpenCV installation
Go to the following link to install OpenCV: https://docs.opencv.org/3.1.0/d7/d9f/tutorial_linux_install.html


## Execution Guide
To run the code, browse to the folder that contains the code and use the following command on Linux Terminal
$ python LBP.py
