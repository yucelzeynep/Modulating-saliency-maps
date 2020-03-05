# Modulating-saliency-maps
This repository contains two zip archives, one image data set and one gaze data set.
The data sets does not contain any demographic or identifying information regarding the participants. 

**Image data set**

The zip archive images.zip contains four folders, ech of which is a set of 8 images of graspable objects depicted over plain white background. The images are collected from the Internet and include objects with different sorts of functional/manipulative characteristics.

In describing functional/manipulative characteristics, we rely on definitions by Natraj et al. (2015). Namely, in interpreting tool affordances, the part where humans grasp and operate an object is typically
termed as its functional part. On the other hand, the end-effector, where the object realizes its purpose, is often termed as manipulative part. For instance, for a knife, the handle is the functional part and the blade is the manipulative part. 

As for object types, we consider the following four types. Specifically, we first distinguish two common forms of functional parts as bar-type and loop-type and build one set of images for each type. In addition, another set of images is dedicated to objects without any explicit functional part. Note that, although these object do not have a dedicated handle or grip, they may have with an intuitive location for grasping. The objects are usually containers such as jar, bottle, paper cup etc.  Moreover, a set of vegetable images is collected for representing non-tool graspable objects. 

Diffferent object types are collected under different folders {b, l, w, x}, where functional/manipulative characteristics are as follows.

 b: objects with *bar*-type grip   
 
 l:  objects with *loop*-type grip  
 
 w:  objects *without* any explicit grip   
 
 x:   non-tools (i.e. vegetables)  

It is demonstrated through a survey that viewers have quite similar ratings regarding their familiarity to the objects irrespective of their age or gender, and there is no peculiar object in the set.

**Gaze recording data set**

The zip archive gaze.zip contains gaze logs of 31 participants, each of which is a separate folder. 

The folder names are organized as YYYY_MM_DD_HH_A_M, where YYYY_MM_DD_HH represents the year, month, day and hour of the recording, A stands for age range and can be either e for 'elderly' and y for 'young adult', and M represents motivation and can be either f for 'free-view', u for 'use' and 'p' for push.

In each folder there are 5 files, 1 file for gaze location estimations and 4 time logs.

The gaze recording file is named as YYYY_MM_DD_HH_MM_SS_mm_gaze.txt, where YYYY_MM_DD_HH is same as above and MM_SS_mm denote the minute, second and millisecond of the recording.

The gaze data in is organized in 3 columns,  where the 1st and 2nd columns include the x- and y- pixels of the estimated gaze location, and the last column is the Unix time stamp in milliseconds. 

The four time log files include the Unix time stamps in milliseconds for the times, when the images are displayed on the screen and removed from the screen. Specifically, the files are named as YYYY_MM_HH_MM_SS_timelog_T.txt, where  YYYY_MM_HH_MM_SS is same as above and T represents object type, which can be either l for loop-type grip, b for bar-type grip, w for no explicit grip and x for non-tools.

In time log files, there is an 8-line header, explaining the organization of the contents. Namely, 1st column includes name of the image file. 2nd column include the time the object is displayed on the screen, and the 3rd column includes the time the image is removed from the screen. Four sets of images depicting objects different functional/manipulative characteristics

**References**

Natraj, N., Pella, Y., Borghi, A., and Wheaton, L. (2015). The visual encoding of tool–object affordances.
Neuroscience 310, 512–527
