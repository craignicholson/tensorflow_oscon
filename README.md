# tensorflow_oscon
Repo to hold all my tensorflow training


# Tensorflow setup for OSOC 2017

## Windows
Got ot this website and download python 3.5

Get this installer...
Click Advanced on install and change the directory so we can easily setup the environmental paths.

C:\Python35

Install...

System Properties

Advanced
Environment Variables at the bottom
System Variables
    Path
        Edit
            New
                C:\Python35
                C:\Python35\Scripts 


Test via the console
python --v

upgrade pip
python -m pip isntall --upgrade pip

Does my windows machine have a GPU?

install tensor flow

check if numpy and scipy is installed

yaml

HDf5 nad h5py for saving and loading model functions

pip3 install 



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


### Installing Tensorflog
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

cn ~ $ python
>>> import tensorflow as tf
>>> tf.__version__
'1.1.0-rc1'

## Windows Setup