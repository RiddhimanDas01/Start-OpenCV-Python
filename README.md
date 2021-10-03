# Start-OpenCV-Python

# Beginner Track On Python with OpenCV

![WhatsApp Image 2021-10-02 at 7 10 41 PM](https://user-images.githubusercontent.com/73641722/135733669-b743db6a-8f77-43ee-ab48-74e0da8715ae.jpeg)


# **INSTALLATION AND SETUP *-***

***Installing Python - [https://youtu.be/aN6OVm0mTHo](https://youtu.be/aN6OVm0mTHo)***

***Installing and setting up OpenCV** - **https://youtu.be/wSkvAKK4oCM***

***Getting Started with JupyterLab -  https://youtu.be/3C9E2yPBw7s***

**Overview of using the *Anaconda prompt* and setting the *path folder* to open *Jupyter Notebook* -**




![mumba](https://user-images.githubusercontent.com/73641722/135733679-6f86c9c6-9c09-4d77-93ea-cfc03e36f526.png)


Here above , 

- *The first command indicates about changing the path directory.*
- *Second , about how to setting up the path folder for opening the jupyter notebook.*
- *Thirdly , the notebook on saving (.ipynb) file gets saved into the folder.*

## ***LEVEL - 1***

**Introduction to Python with OpenCV**

***OpenCV*** is a Python open source library used for computer vision in Artificial intelligence, Machine Learning, face recognition, face detection etc.It is used to develop real-time computer vision applications. It is capable of processing images and videos to identify objects, faces, or even handwriting.

Python -

Python is a powerful general-purpose programming language. It is used in web development, data science, creating software prototypes, and so on. Fortunately for beginners, Python has simple easy-to-use syntax.

**Flow Control -**

Loops and Conditional statements **-**

**Conditional Statements , Refer to -**

**[https://www.programiz.com/python-programming/if-elif-else](https://www.programiz.com/python-programming/if-elif-else)**

**Looping Structures , Refer to -**

**For Loop : [https://www.programiz.com/python-programming/for-loop](https://www.programiz.com/python-programming/for-loop)**

**While Loop : [https://www.programiz.com/python-programming/while-loop](https://www.programiz.com/python-programming/while-loop)**

**Alteration of Loop Statements : [https://www.programiz.com/python-programming/break-continue](https://www.programiz.com/python-programming/break-continue)**

**OOP in Python -**

****Python is a multi-paradigm programming language. It supports different programming approaches.One of the popular approaches to solve a programming problem is by creating objects. This is known as Object-Oriented Programming (OOP).

**Refer , [https://www.programiz.com/python-programming/object-oriented-programming](https://www.programiz.com/python-programming/object-oriented-programming)**

### **Tasks : *attach the .ipynb file and make a pull request***

**A Quick Brush-up -**

- **Accept a list of 9 float numbers as an input from the user**
- **Write a program to display all prime numbers within a range**
- Create a **van** child class that inherits from the **Vehicle** The default fare charge of any vehicle is **seating capacity * 55** . If **Vehicle** is **van** instance, we need to add an extra 8% on full fare as a maintenance charge. So total fare for **van** instance will become the **final amount = total fare + 8% of the total fare.**

## ***LEVEL - 2***

**Library , Modules and Functions**

***Numpy -***

It is a scientific computation package It offers many functions and utilities to work with N-Dimension arrays Largely used by other libraries such as OpenCV, TensorFlow and PyTorch to deal with multi dimensional arrays (e.g., tensors or images).

***OpenCV -***

OpenCV is an open source Computer Computer Vision library. It allows to develop complex Computer Vision and Machine Learning applications fast, offering a wide set of functions. Originally developed in C/C++, now OpenCV has handlers also for Java and Python it can be exploited also in IOS and Android apps. In Python, OpenCV and NumPy are strictly related.

**Link for Material** - [https://drive.google.com/file/d/1ipqiecHbs6Vr1asnyEqt378lmwl46YO2/view?usp=sharing](https://drive.google.com/file/d/1ipqiecHbs6Vr1asnyEqt378lmwl46YO2/view?usp=sharing)

***Pandas -***

pandas is a Python package providing fast, flexible, and expressive data structures designed to make working with “relational” or “labeled” data both easy and intuitive. It aims to be the fundamental high-level building block for doing practical, **real-world** data analysis in Python.

**Document preference - [https://pandas.pydata.org/docs/reference/index.html](https://pandas.pydata.org/docs/reference/index.html)**

***Matplotlib -***

The pyplot API has a convenient MATLAB-style stateful interface. In fact, matplotlib was originally written as an open source alternative for MATLAB. The OO API and its interface is more customizable and powerful than pyplot, but considered more difficult to use. As a result, the pyplot interface is more commonly used, and is referred to by default in this article.

Understanding matplotlib’s pyplot API is key to understanding how to work with plots:

- ***pyplot.figure: Figure***is the top-level container. It includes everything visualized in a plot including one or more ***Axes***.
- ***pyplot.axes**:**Axes*** contain most of the elements in a plot**: *Axis, Tick, Line2D, Text,*** etc., and sets the coordinates. It is the area in which data is plotted. Axes include the X-Axis, Y-Axis, and possibly a Z-Axis, as well.

**Numpy** is a package for scientific computing. Numpy is a required dependency for **matplotlib**, which uses **numpy** functions for numerical data and multi-dimensional arrays.

**Pandas** is a library used by matplotlib ****mainly for data manipulation and analysis. Pandas provides an in-memory 2D data table object called a Dataframe. Unlike numpy, pandas is not a required dependency of matplotlib.

**Document preference - [https://matplotlib.org/stable/tutorials/introductory/pyplot.html](https://matplotlib.org/stable/tutorials/introductory/pyplot.html)**

## *LEVEL-3*

**More about dataset , Predefined models and Parameters**

***COCO DATASET*** -

**The MS COCO (Microsoft Common Objects in Context)** dataset is a large-scale object detection, segmentation, key-point detection, and captioning dataset. The dataset consists of 328K images.

The dataset has annotations for-

- **Object detection**: bounding boxes and per-instance segmentation masks with 80 object categories.
- **Captioning**: natural language descriptions of the images (see MS COCO Captions),
- **Keypoints detection**: containing more than 200,000 images and 250,000 person instances labeled with keypoints (17 possible keypoints, such as left eye, nose, right hip, right ankle),
- **Stuff image segmentation**– per-pixel segmentation masks with 91 stuff categories, such as grass, wall, sky (see MS COCO Stuff),
- **Panoptic**: full scene segmentation, with 80 thing categories (such as person, bicycle, elephant) and a subset of 91 stuff categories (grass, sky, road),
- **Dense pose**: more than 39,000 images and 56,000 person instances labeled with DensePose annotations – each labeled person is annotated with an instance id and a mapping between image pixels that belong to that person body and a template 3D model. The annotations are publicly available only for training and validation images.

***To Know more - [https://towardsdatascience.com/how-to-work-with-object-detection-datasets-in-coco-format-9bf4fb5848a4](https://towardsdatascience.com/how-to-work-with-object-detection-datasets-in-coco-format-9bf4fb5848a4)***

***SSD MobileNet Architecture -***

The SSD architecture is a single convolution network that learns to predict bounding box locations and classify these locations in one pass. Hence, SSD can be trained end-to-end.

By using SSD, we only need to **take one single shot to detect multiple objects within the image**, while regional proposal network (RPN) based approaches such as R-CNN series that need two shots, one for generating region proposals, one for detecting the object of each proposal. Thus, SSD is much faster compared with two-shot RPN-based approaches.

For more details about SSD architecture and its working [here](https://arxiv.org/pdf/1512.02325.pdf)

**TensorFlow object detection API** is the framework for creating a deep learning network that solves object detection problems.There are already pretrained models in their framework which they refer to as Model Zoo. This includes a collection of pretrained models trained on the COCO dataset, the KITTI dataset, and the Open Images Dataset.model

They are also useful for initializing your models when training on the novel dataset. The various architectures used in the pretrained model are described in this table

For more about TensorFlow object detection API, visit their github repo [here](https://github.com/tensorflow/models/tree/master/research/object_detection).

*For modes of implementation of ***TensorFlow object detection API *-* [https://docs.google.com/document/d/1OZzFTliHCMBvXhs5w_G1396Wa8RGXJFZ/edit?usp=sharing&ouid=112413800539249462468&rtpof=true&sd=true](https://docs.google.com/document/d/1OZzFTliHCMBvXhs5w_G1396Wa8RGXJFZ/edit?usp=sharing&ouid=112413800539249462468&rtpof=true&sd=true)**

## *LEVEL - 4*

**Working with Images and Videos**

### *How to read an Image and perform operations on static images -*

- Import the module from the libraries
- Using library method and functions
- Extraction and displaying the required parameters on the image

![Screenshot 2021-10-01 021635](https://user-images.githubusercontent.com/73641722/135733732-80cc1889-8cee-4150-8f9c-8d7764d60856.png)


![1](https://user-images.githubusercontent.com/73641722/135733741-4f4339db-9df3-48df-a93e-3e4a1e3eeb58.png)


### Tasks : *attach the .ipynb file and make a pull*

- **To read ,show and plot an Image .**
- **To read , show and save an image -**
- [ ] ***Open in a grayscale mode , further testing operations ( convert it from the BGR to the RGB format ) and plot the image.***

![Screenshot 2021-10-01 025151](https://user-images.githubusercontent.com/73641722/135733720-92f8a736-7abe-4e2c-814e-7b66ab55aede.png)


### Tasks : ***attach the .ipynb file and make a pull request***

- **To read ,show and plot the first image and the second image -**
- [ ]  ***Further operations , subtract both the images -> convert the result image from original format to RGB format -> plot the resultant image.***
- **To perform Bitwise AND and OR between two input images .**
- **To perform color filtering in an image using the import of appropriate module.**

---

For a better clarification and reference - **refer to this link -> [https://www.geeksforgeeks.org/filter-color-with-opencv/](https://www.geeksforgeeks.org/filter-color-with-opencv/)**

**Refer to this link - [https://likegeeks.com/python-image-processing/](https://likegeeks.com/python-image-processing/)**

- **Save the Image (.jpg) in the set path before performing operations on the image.**

### ***How to work with the moving images and video -***

- Import the module from the libraries
- Using library method and functions

![vid](https://user-images.githubusercontent.com/73641722/135733747-5f306e24-8d35-4e27-8f03-00d3e6efdb6b.png)


### Tasks : ***attach the .ipynb file and make a pull request***

- **To capture and return the frames of video using the system camera .**
- **To read and display the video -**
- [ ]  ***Open in a grayscale mode , further testing operations ( convert it from the BGR to the RGB format ) and plot the image.***

**Refer to this link -** [https://learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/](https://learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/)

- **Save the Video (.mp4/.avi) in the set path before performing operations on the video.**

## ***LEVEL-5***

**Application and Project**

# PROJECT -

# **1.**

## ***OBJECT DETECTION :***

### **Implementing an object detector which identifies the classes of objects in any image or a video.**

- Set path should contain the images and videos being used during the execution of the code.
- All the predefined models , trained graphs and preferred Data set should be initialized in the code through the path.

**Frozen Graph - [https://drive.google.com/file/d/1UN4d4E5ICyiVltgoJlOD_qr_mUVBYEME/view?usp=sharing](https://drive.google.com/file/d/1UN4d4E5ICyiVltgoJlOD_qr_mUVBYEME/view?usp=sharing)**

**Model Configuration - [https://drive.google.com/file/d/1e_WeDoTxLNa0KR-kBKb7WkgJxjRR72ph/view?usp=sharing](https://drive.google.com/file/d/1e_WeDoTxLNa0KR-kBKb7WkgJxjRR72ph/view?usp=sharing)**


![7](https://user-images.githubusercontent.com/73641722/135733754-c053b822-01fa-4d1f-a6b1-a3b91613b3db.png)

![9](https://user-images.githubusercontent.com/73641722/135733757-3fc96483-d214-4a72-925a-0c12ce666454.png)

![0](https://user-images.githubusercontent.com/73641722/135733767-961836e2-912e-4735-80ac-aaf4090f87b3.png)



# **2.**

## ***COLOR DETECTION :***

### Implementing an image color detector which identifies the colors in any image .

**Colors Dataset - [https://github.com/codebrainz/color-names/blob/master/output/colors.csv](https://github.com/codebrainz/color-names/blob/master/output/colors.csv)**

**CSV File (Sample) - [https://drive.google.com/file/d/1qJz88d4eLZxvhZrOJixPY5y1pd48_k3m/view?usp=sharing](https://drive.google.com/file/d/1qJz88d4eLZxvhZrOJixPY5y1pd48_k3m/view?usp=sharing)**

- Set path should contain the images and videos being used during the execution of the code.
- All the predefined models , trained graphs and preferred Data (if used) ,  CSV files set should be initialized in the code through the path.

![111](https://user-images.githubusercontent.com/73641722/135733816-80f242d5-1bad-403e-87c5-cd7ed6a5a723.png)

![112](https://user-images.githubusercontent.com/73641722/135733818-4b41a7df-9f7d-4e33-a9d6-a6a0716f6a73.png)

![113](https://user-images.githubusercontent.com/73641722/135733820-96d0e853-f175-48e8-a99d-83a8f218825a.png)



