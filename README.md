# *Keras implementation*

## Explanation

This is a working Keras CNN from a Senior Project that I Carried in Spring of 2022. Since I hadn't known how to write CNNs in Keras prior to this 
project, so I started from an example I found for cat/dog image classification that I had found on the internet. This google collab is linked here:
https://colab.research.google.com/drive/1ZfablQK6hTqG4X2v3KrqJUj9iLtbZH_1

It's a different project, but the fundamental components of it, along with studying Keras documentation gave me enough direction
to write the FloorNet.py file in this repo. The network architecture was inspired from an older paper and repo linked below. The paper and repo itself was 
very old TF 1.0 code, and we were told to modernize it. I wrote the Following Keras implementation on my own. My groupmates were overwhelmed with
other classes at the time and weren't too keen on helping once I got some footing in the project. 

paper: https://arxiv.org/pdf/1908.11025.pdf
repo: https://github.com/zlzeng/DeepFloorplan

## Prerequisite

  - Dataset Download Link
  - https://www.dropbox.com/sh/9yi1p4evb0ogxlu/AABsWU73c-YQCHZKayl_dzj7a?dl=0
  - Model Download Link
  - https://www.dropbox.com/scl/fo/1phah9bm81xists8kl17m/h?dl=0&rlkey=5wyddudw9eqsrzoapz2c9rmnq

  Make a new file for the Dataset Folders and call it "dataset". Remove the jp file from the variables listed below. Have the Model files and the FloorplanNet.py in the same folder. House the Dataset and (Net and Models) inside the same directory.

## Required packages

  ```python
  - TensorFlow [2.8]
  - Numpy [latest]
  - Keras [latest]
  - Mathplotlib [latest]
  - Pathlib [latest]
  - OpenCV [latest]
  - PIL [latest]
  ```

### Operting System

**Windows/Mac OSX**

> VS Code

**Linux**

> Terminal

### Notes

Before run it, there are few varaibles that need to be changed depending on your dataset location if you are using other dataset. 

The related variables are:

- `new_path_wall`
- `new_path_close`
- `nb_images`
- `tf.data.Dataset.list_files`

## Training models

- In order to train the models, the `CUDA` version `11.6` is required to be installed on the machine and `NVIDIA` graphics card is required for this functionality.

- Our models were built with a `GTX 1660`, which hits the bare minimum to train the models

- Recommended to have a newer `NVIDIA` graphic card in order to train the models more efficiently

## Code explaination

  - `FloorplanNet.py`: We integrate everything under one file, including handling the input and output with tfrecords. We firstly manually import the graphic card toolkit folder so that we are able to use it during training. 
  -  `load_raw_images` load all of the images and get the number of each image based on their paths. 
  -  `to_tfrecord` transfer image into tfrecord format so that it can be easily trained with tensorflow and keras. Also, they load the wall label images to tfrecord. 
  -  `read_tfrecord` function read the images and labels into float 32 to make CNN handled easily. 
  -  `get_batched_dataset` is applying tfrecord into datasets for training. In the main, it firstly transfers all data into tfrecord and then uses LeakyReLU activation function. There are about 5 different set of layers that includes different filter number in the model, this increases the resolution and accuracy of the training result. From line 204 to 258, it loads the model for walls, boundarys and rooms. The rest of lines are displaying the output.
  