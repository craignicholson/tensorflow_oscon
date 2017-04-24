# Tensorflow setup for OSOC 2017
Repo to hold all my tensorflow training


## Dependencies
* git
* tensorflow
    python 3.5 on windows
    python 2.7 mac

* keras
    * scipy
    * numpy
    * pyymal


## Pretrained models

Pretrained models (we recommend downloading in advance before the training begins)

[Pretrained vision model](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz) [Pretrained GloVe word embeddings](http://nlp.stanford.edu/data/glove.42B.300d.zip)

## Windows

### Install Python 3.5, which works for tensorflow 1.1

Download python 3.5

* https://www.python.org/downloads/windows/
* python-3.5.2-amd64.exe

Click Advanced on install and change the directory so we can easily setup the  environmental paths.

Set the directory to C:\Python35 (Or your choice)

Install...

#### Add Environment Variables
On your Windows PC go to My computer -> right click anywhere and select Properties -> Advanced system setting -> Environment variables
** Environment Variables is at the bottom **

System Variables

* Path (Choose the Path Variable)
  * Edit
    * New (Add the new paths for python, and .\scripts for pip)
      * C:\Python35
      * C:\Python35\Scripts

#### Test Environment Variable
Open powershell or command prompt

> C:\>python --version

```bash

Python 3.5.2

```

Update pip
> C:\> python -m pip install --upgrade pip
> C:\>pip3 --version

```bash

pip 9.0.1 from c:\python\python35\lib\site-packages (python 3.5)

```

### Install tensorflow

> C:\>pip3 install --upgrade tensorflow

```bash

Collecting tensorflow
  Downloading tensorflow-1.1.0-cp35-cp35m-win_amd64.whl (19.4MB)
    100% |################################| 19.4MB 54kB/s
Collecting wheel>=0.26 (from tensorflow)
  Downloading wheel-0.29.0-py2.py3-none-any.whl (66kB)
    100% |################################| 71kB 1.5MB/s
Collecting six>=1.10.0 (from tensorflow)
  Using cached six-1.10.0-py2.py3-none-any.whl
Collecting protobuf>=3.2.0 (from tensorflow)
  Using cached protobuf-3.2.0-py2.py3-none-any.whl
Collecting werkzeug>=0.11.10 (from tensorflow)
  Downloading Werkzeug-0.12.1-py2.py3-none-any.whl (312kB)
    100% |################################| 317kB 1.1MB/s
Collecting numpy>=1.11.0 (from tensorflow)
  Using cached numpy-1.12.1-cp35-none-win_amd64.whl
Collecting setuptools (from protobuf>=3.2.0->tensorflow)
  Downloading setuptools-35.0.1-py2.py3-none-any.whl (390kB)
    100% |################################| 399kB 779kB/s
Collecting appdirs>=1.4.0 (from setuptools->protobuf>=3.2.0->tensorflow)
  Downloading appdirs-1.4.3-py2.py3-none-any.whl
Collecting packaging>=16.8 (from setuptools->protobuf>=3.2.0->tensorflow)
  Downloading packaging-16.8-py2.py3-none-any.whl
Collecting pyparsing (from packaging>=16.8->setuptools->protobuf>=3.2.0->tensorflow)
  Downloading pyparsing-2.2.0-py2.py3-none-any.whl (56kB)
    100% |################################| 61kB 1.8MB/s
Installing collected packages: wheel, six, appdirs, pyparsing, packaging, setuptools, protobuf, werkzeug, numpy, tensorflow
  Found existing installation: setuptools 20.10.1
    Uninstalling setuptools-20.10.1:
      Successfully uninstalled setuptools-20.10.1
Successfully installed appdirs-1.4.3 numpy-1.12.1 packaging-16.8 protobuf-3.2.0 pyparsing-2.2.0 setuptools-35.0.1 six-1.10.0 tensorflow-1.1.0 werkzeug-0.12.1 wheel-0.29.0

```

### Test tensorflow

> C:\>python

```bash

Python 3.5.2 (v3.5.2:4def2a2901a5, Jun 25 2016, 22:18:55) [MSC v.1900 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
2017-04-24 11:19:51.311260: W c:\tf_jenkins\home\workspace\release-win\device\cpu\os\windows\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE instructions, but these are available on your machine and could speed up CPU computations.
2017-04-24 11:19:51.312417: W c:\tf_jenkins\home\workspace\release-win\device\cpu\os\windows\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE2 instructions, but these are available on your machine and could speed up CPU computations.
2017-04-24 11:19:51.313051: W c:\tf_jenkins\home\workspace\release-win\device\cpu\os\windows\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE3 instructions, but these are available on your machine and could speed up CPU computations.
2017-04-24 11:19:51.313691: W c:\tf_jenkins\home\workspace\release-win\device\cpu\os\windows\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2017-04-24 11:19:51.314391: W c:\tf_jenkins\home\workspace\release-win\device\cpu\os\windows\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-04-24 11:19:51.315030: W c:\tf_jenkins\home\workspace\release-win\device\cpu\os\windows\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2017-04-24 11:19:51.315739: W c:\tf_jenkins\home\workspace\release-win\device\cpu\os\windows\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2017-04-24 11:19:51.316386: W c:\tf_jenkins\home\workspace\release-win\device\cpu\os\windows\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
>>> print(sess.run(hello))
b'Hello, TensorFlow!'
>>> print(sess.run(hello))
b'Hello, TensorFlow!'
>>> tf.__version__
'1.1.0'
>>> quit()

```

### Install numpy for python 3.5

Download numpy to some directory...

http://www.lfd.uci.edu/~gohlke/pythonlibs/

numpy‑1.12.1+mkl‑cp35‑cp35m‑win_amd64.whl

> C:\>pip3 install "numpy-1.12.1+mkl-cp35-cp35m-win_amd64.whl"
Processing c:\numpy-1.12.1+mkl-cp35-cp35m-win_amd64.whl
Installing collected packages: numpy
  Found existing installation: numpy 1.12.1
    Uninstalling numpy-1.12.1:
      Successfully uninstalled numpy-1.12.1
Successfully installed numpy-1.12.1+mkl

### Install scipty for python 3.5
http://www.lfd.uci.edu/~gohlke/pythonlibs/

scipy-0.19.0-cp35-cp35m-win_amd64.whl

> C:\>pip3 install "scipy-0.19.0-cp35-cp35m-win_amd64.whl"

```bash

Processing c:\scipy-0.19.0-cp35-cp35m-win_amd64.whl
Requirement already satisfied: numpy>=1.8.2 in c:\python\python35\lib\site-packages (from scipy==0.19.0)
Installing collected packages: scipy
Successfully installed scipy-0.19.0

```

### Test numpy, scipy, yaml

> C:\>python

```python

Python 3.5.2 (v3.5.2:4def2a2901a5, Jun 25 2016, 22:18:55) [MSC v.1900 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import scipy
>>> import numpy
>>> scipy.__version__
'0.19.0'
>>> numpy.__version__
'1.12.1'
>>> import yaml
>>> yaml.__version__
'3.12'
>>> quit()

```

### Install Keras

> C:\>pip3 install --upgrade keras

```bash

Collecting keras
  Downloading Keras-2.0.3.tar.gz (196kB)
    100% |################################| 204kB 900kB/s
Collecting theano (from keras)
  Downloading Theano-0.9.0.tar.gz (3.1MB)
    100% |################################| 3.1MB 318kB/s
Requirement already up-to-date: pyyaml in c:\python\python35\lib\site-packages (from keras)
Requirement already up-to-date: six in c:\python\python35\lib\site-packages (from keras)
Requirement already up-to-date: numpy>=1.9.1 in c:\python\python35\lib\site-packages (from theano->keras)
Requirement already up-to-date: scipy>=0.14 in c:\python\python35\lib\site-packages (from theano->keras)
Building wheels for collected packages: keras, theano
  Running setup.py bdist_wheel for keras ... done
  Stored in directory: C:\Users\Craig Nicholson\AppData\Local\pip\Cache\wheels\93\72\65\4924fd6b1859343291c50774e2df36919ee61c4511dc6a9890
  Running setup.py bdist_wheel for theano ... done
  Stored in directory: C:\Users\Craig Nicholson\AppData\Local\pip\Cache\wheels\d5\5b\93\433299b86e3e9b25f0f600e4e4ebf18e38eb7534ea518eba13
Successfully built keras theano
Installing collected packages: theano, keras
Successfully installed keras-2.0.3 theano-0.9.0

```

### Test Keras

C:\>python

```python

Python 3.5.2 (v3.5.2:4def2a2901a5, Jun 25 2016, 22:18:55) [MSC v.1900 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import keras; keras.__version__
Using TensorFlow backend.
'2.0.3'
>>>

```

### Install other stuff pyaml, scikit-learn, matplotlib

> C:\>pip3 install --upgrade pyyaml

```bash

Collecting pyyaml
  Downloading PyYAML-3.12-cp35-cp35m-win_amd64.whl (195kB)
    100% |################################| 204kB 732kB/s
Installing collected packages: pyyaml
Successfully installed pyyaml-3.12

```

> C:\>pip3 install --upgrade scikit-learn

```bash

Collecting scikit-learn
  Downloading scikit_learn-0.18.1-cp35-cp35m-win_amd64.whl (4.1MB)
    100% |################################| 4.1MB 240kB/s
Installing collected packages: scikit-learn
Successfully installed scikit-learn-0.18.1

```

> C:\>pip3 install --upgrade matplotlib

```bash

Collecting matplotlib
  Using cached matplotlib-2.0.0-cp35-cp35m-win_amd64.whl
Collecting cycler>=0.10 (from matplotlib)
  Using cached cycler-0.10.0-py2.py3-none-any.whl
Requirement already up-to-date: six>=1.10 in c:\python\python35\lib\site-packages (from matplotlib)
Requirement already up-to-date: numpy>=1.7.1 in c:\python\python35\lib\site-packages (from matplotlib)
Requirement already up-to-date: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=1.5.6 in c:\python\python35\lib\site-packages (from matplotlib)
Collecting python-dateutil (from matplotlib)
  Using cached python_dateutil-2.6.0-py2.py3-none-any.whl
Collecting pytz (from matplotlib)
  Using cached pytz-2017.2-py2.py3-none-any.whl
Installing collected packages: cycler, python-dateutil, pytz, matplotlib
Successfully installed cycler-0.10.0 matplotlib-2.0.0 python-dateutil-2.6.0 pytz-2017.2

```

### HDF5 or h5py

What are these... Keras might need these for oscon.  Review it.
HDf5 nad h5py for saving and loading model functions

## Mac Setup

### Installing Python

    Install Homebrew https://brew.sh/
    brew install python

### Installing Pip

Anaconda was already setup... from an older Continuium Download
* Python 2.7.12 :: Anaconda custom (x86_64)

cn ~ $  pip -V
        pip 8.1.1 from /Users/cn/anaconda/lib/python2.7/site-packages (python 2.7)
cn ~ $  sudo easy_install --upgrade pip
        Best match: pip 9.0.1

cn ~ $  pip -V
        pip 9.0.1 from /Users/cn/anaconda/lib/python2.7/site-packages/pip-9.0.1-py2.7.egg (python 2.7)


### Installing Tensorflow
https://www.tensorflow.org/install/install_mac

cn ~ $  sudo pip uninstall tensorflow
cn ~ $  python 
        Python 2.7.12 |Anaconda custom (x86_64)| (default, Jul  2 2016, 17:43:17) 
        [GCC 4.2.1 (Based on Apple Inc. build 5658) (LLVM build 2336.11.00)] on darwin
        Type "help", "copyright", "credits" or "license" for more information.
        Anaconda is brought to you by Continuum Analytics.
        Please check out: http://continuum.io/thanks and https://anaconda.org
        
        >>> import tensorflow as tf
        >>> hello = tf.constant('Hello, TensorFlow!')
        >>> sess = tf.Session()

W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, 
    but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, 
    but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, 
    but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, 
    but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, 
    but these are available on your machine and could speed up CPU computations.
>>> print(sess.run(hello))
Hello, TensorFlow!

cn ~ $  python -c 'import tensorflow as tf; print(tf.__version__)'  # for Python 2
        1.0.1

### Install Keras
cn ~ $ pip install keras
cn ~ $ pip install --upgrade keras

cn ~ $ mkdir -p ~/.keras
cn ~ $ echo '{"epsilon":1e-07,"floatx":"float32","backend":"tensorflow"}' > ~/.keras/keras.json

Where did this get created?

### Make Sure Keras runs
cn ~ $ curl -sSL https://github.com/fchollet/keras/raw/master/examples/mnist_mlp.py | python


https://ermaker.github.io/blog/2016/06/22/get-started-with-keras-for-beginners-tensorflow-backend.html


### Models for the class.
http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz


bumb version up

cn ~ $ python
>>> import tensorflow as tf
>>> tf.__version__
'1.0.1'

sudo pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.1.0rc1-py2-none-any.whl

> cn ~ $ python

```bash

>>> import tensorflow as tf
>>> tf.__version__
'1.1.0-rc1'

```