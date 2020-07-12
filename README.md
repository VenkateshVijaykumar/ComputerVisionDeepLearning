# ComputerVisionDeepLearning
A collection of implementations and applications of CNN architectures, the repository contains a collection of commonly encountered architectures for deep learning based image classifiers. The architectures are implemented in a modular form, so as to avoid repeated network building for every application. 

The architectures implemented are:

 - GoogLeNet
 - LeNet
 - ShallowNet
 - VGGNet (Mini)

The implementations are in plain `Keras`, with a `TensorFlow 1.x.x` backend. These can easily be migrated to `TF2.0` and `tf.keras` with a few minor tweaks.

Further implemented are learning rate decay schedulers, both from within the standard keras implementation, and with a custom stepwise (or piecewise) decay function. Monitoring and checkpointing the training progress has also been implemented in this repo. Data augmentation using the Keras standard image data generator has also been performed and evaluated. Furthermore, feature extraction using these networks has also been performed, and the resulting features have been used to implement a linear SVM classifier. Finally, network fine-tuning using a pre-trained base network, with a trainable "head" transplanted has also been implemented.

Requirements for running the code in this repo are:

 - `Keras`
 - `TensorFlow 1.x.x`
 - `SkLearn`
 - `NumPy`
 - `Matplotlib`

The repository has been tested on the [Google Colab](https://colab.research.google.com/) environment, and has reported expected results. Owing to testing on this environment, the associated `iPython Notebooks` are also included in the repository. Further, owing to the characteristics of the Colab environment, it was necessary to add the relevant paths pointing to the modular Python files to the `OS PATH` variable, if using on a local machine, these lines can be commented out from the top of the application files.

> [UPDATE 7-12-2020]
> Added TensorFlow 2 versions, more specifically updated versions of the networks using `tensorflow.keras`
> To use, simply replace the existing architectures in the `PPM\nn\conv` directory with the files in `TF2_Versions`. Alternatively, these may be called as standalone classes from the `TF2_Versions` directory itself for implementations that do not involve the auxiliary modules from the `PPM` directory.
