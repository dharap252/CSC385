# Dhara Patel
### _Progress Log_
### _CSC 385_
-------
## Week 2 – Choice Week (January 30th, 2021)

The initial choice was to continue my focus on studying robotics and implement a project idea I had been wanting to try out. However, the project required in depth knowledge of various artificial intelligence tools such as deep learning, reinforced learning, and computer vision/image processing. After thinking through I believe computer vision will be the most appropriate topic to explore for the duration of the Spring 2021 semester. Advanced computer vision requires application of deep learning and reinforced learning can also be applied to attempt to get a step closer to achieving human vision capabilities.

Computer vision is a subfield machine learning and artificial intelligence that focuses on attempting to replicate the human vision, enabling computer to identify and process objects and situations in image and videos. I would like to learn to implement and apply computer vision algorithms to autonomous driving, but the time restrictions only allow me to explore the very basics of computer vision of which semantic image segmentation covers a quicker and complex algorithm for recognizing and understanding what is in the image on pixel level and identify the object.

-------
## Week 3 – Research Report Choice Week (February 6th, 2021)

To read any articles, thesis papers, or conference papers on any topic I use ResearchGate. There are various articles and papers of different complexity to refer to. I have listed few of the articles I have read and gained an understanding from. Overall, I realized that even semantic segmentation alone is very complex to understand on theoretical level and requires in-depth understanding of mathematics and coding language of choice. Python combined with PyTorch or TensorFlow seem to be a very popular choice of language and tool to implement semantic image segmentation. I will need to do more research in the upcoming week to think through and conclude on exactly where I want to start and what I want to do.
##### [Research Papers](#)
- https://www.researchgate.net/publication/326012274_Recent_progress_in_semantic_image_segmentation
- https://www.researchgate.net/publication/342171446_Introspective_Failure_Prediction_for_Semantic_Image_Segmentation
- https://www.researchgate.net/publication/261204452_A_benchmark_for_semantic_image_segmentation
- https://www.researchgate.net/publication/314237740_Adversarial_Examples_for_Semantic_Image_Segmentation

-------
## Week 5 – Specifications and Timeline (February 21st, 2021)

I am currently on at the point in my research where I can identify which data structure I would like to use. I have narrowed it down to using semantic image segmentation for processing a single image instead of a full video stream. I am not sure as to what complexity level I will be able to reach in image processing by end of this project but nevertheless it a progress in the direction of having a small understanding of how computer vision works and how to go about implementing it. 

I think the language most suited for learning image processing is python, but I am not sure as to which tool would be best suited between TensorFlow, PyTorch, and Keras. I believe by end of week 6 I will have a clear idea of the flow of my project and have my design plan “finalized” for at least the starting stage of my project. After the design plan is completed, I would be able to provide a timeline for my project. 

-------
## Week 7 – Deeper Understanding
Getting more indepth as to how semantic segmentation works and methodologies behind it
##### [WATCHED](#)
- [Semantic Segmentation Tutorial | Training a Semantic Segmentation Network | Great Learning](https://www.youtube.com/watch?v=nuO926-RLQI&ab_channel=GreatLearning)
- [CV3DST - Semantic Segmentation](https://www.youtube.com/watch?v=XMSjOatyH0k&t=47s&ab_channel=DynamicVisionandLearningGroup)
- [Lecture 11 | Detection and Segmentation](https://www.youtube.com/watch?v=nDPWywWRIRo&ab_channel=StanfordUniversitySchoolofEngineering)
- [Thresholding in Image Processing](https://www.youtube.com/watch?v=vtbdqq7yAcc&ab_channel=LearningOrbis)

-------
## ✨ Week 8 – Research Report ✓
Review Notes, Reread the two articles, and write the report

-------
## Week 9 – Deep Leaning Focus and LEARN USE OF OpenCV
I had to step into deep leanrning becasue I couldn't figure out how to approach semantic segmentation without the use of deep leaning.
- [Deep Learning: A Crash Course](https://www.youtube.com/watch?v=r0Ogt-q956I&ab_channel=ACMSIGGRAPH)
- [Image Segmentation Using Deep Learning](https://www.youtube.com/watch?v=hjMwPjU1tRc&ab_channel=DepartmentofComputerEngineering%2CPIET%2CJaipur)

By end of this week I am still having trouble understading and grasping the full concept from the two video I have watched.

##### [OpenCV](#)
- [16 OpenCV Functions to Start your Computer Vision journey (with Python code)](https://www.analyticsvidhya.com/blog/2019/03/opencv-functions-computer-vision-python/)

-------
## Week 10 – Prototype Week
Attempted to move past tutorials and use OpenCV to apply image threashold example to start implementation of semantic segmentation - FAIL

##### [TUTORIAL followed](#)
- [28 - Thresholding and morphological operations using openCV in Python](https://www.youtube.com/watch?v=WQK_oOWW5Zo&ab_channel=DigitalSreeni)

and updated using what I learned from the link listed under OPENCV in ```Week 9```.

-------
## Week 11 – Watershed Algorithm Research
##### [Followed provided tutorial](#)
- https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_watershed/py_watershed.html
##### [Watershed segmentation](#)
-  https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html

-------
## Week 12 – SLIC Algorithm Research

- https://docs.opencv.org/3.4/df/d6c/group__ximgproc__superpixel.html
--> Documentation on superpixel functions
- https://medium.com/@darshita1405/superpixels-and-slic-6b2d8a6e4f08
--> Understanding super pixel and SLIC
- [Python: Superpixel SLIC algorithm used (Tutorial followed)](https://www.programmersought.com/article/46534235498/)
- scikit-image package and subpackages documentation  
--> https://scikit-image.org/docs/dev/api/skimage.html

-------
## Week 13 & 14 – Tutorials & Presentation

##### [Other Tutorials and articles](#)
- [Detailed explanation of SLIC superpixel segmentation (1): Introduction](https://www.programmersought.com/article/34834112462/)
- ["Digital Image Processing" Learn about Superpixel Super pixel introduction Non-deep learning methods](https://www.programmersought.com/article/31754409864/)
- [Segmentation: A SLIC Superpixel Tutorial using Python (Super Pixel (Pixel Grid Classification))](https://www.programmersought.com/article/5054518555/)
- [Super pixel segmentation of image processing](https://www.programmersought.com/article/65176828835/)
- [(Digital image processing 5.3) SLIC algorithm Super pixel segmentation (unsupervised clustering method) python](https://www.programmersought.com/article/88757591839/)

##### [Libraries and functions to learn more about](#)
- [Class implementing the LSC (Linear Spectral Clustering) superpixels.](https://docs.opencv.org/3.4/df/d6c/group__ximgproc__superpixel.html)
```sh
#include <opencv2/ximgproc/slic.hpp>

Ptr<SuperpixelLSC> cv::ximgproc::createSuperpixelLSC (InputArray image, int region_size = 10, float ratio = 0.075f)		

Python:
cv.ximgproc.createSuperpixelLSC(image[, region_size[, ratio]]) -> retval
```
### ✨Presentation  (April 29) ✓

> NOTE:  Found the following link and C++ implementation [Deep optimization and analysis of the code of the superpixel classic algorithm SLIC.](https://www.programmersought.com/article/1494766613/) and will attempt to implement this in C++ then python later on.

-------
## Week 15 – Update Files and Write Review

ReviewReport.md
