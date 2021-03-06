# The Varroa Destructor Destructor Code

## 06 bee mobilenet
The first time running a transfer trained mobilenet to detect bee classes.

## 07 bee stacked nns
Running two networks in series.
The first one detects bees in the frame.
The second network classifies the bees for their health.

## 08 nn comparison
An extension to 07.
Different networks can easily be loaded to quickly compare their results.
The bees now are assigned an unique id and are tracked over multiple frames.

## 10 destruction
Here it is all put together.
Bees are detected and classified by their health.
If at least one infested bee was found (or it is in demonstration mode with -d), the bees position is targeted with SALT.
The System for Advanced Laser Tracking.

## 14 bee tracker
Updated version of the bee tracking.
Instead on relying on some very simple Python code, the DepthAI object tracker is used.
It comprises a calman filter to get a better position estimation of moving objects, in this case the bees.

# Helper code
These are some useful functions or code snippets.

## credentials
Used to store personal login data.
This file is in the gitignore and thus will not be stored online.

## prelabel
My take on a SuperAnnotate clone.
For each image in a source directory a detection network is run.
The detections are stored in an xml file in a format readable by labelImg.
This will reduce the time to label new images, because only corrections will have to be made by hand.

## saveimages
Saves one 1080P image per second.
The filename is the date in the format yyyymmddhhmmss.
The file is encoded as jpeg.

## salt
SALT is the System for Advanced Laser Tracking.
It controlls a servo gimbal and a laser diode.

## 11 save video
Saves video.

## 12 save images
Saves images.

## 13 save images and video
The code used for capturing data.
1080P video is captured at 30 frames per second and encoded as h265.
At the same time individual frames are encoded as jpeg.
Both outputs can be saved.
To save disk space, only every 30th frame is stored, meaning one frame per second.

# Testing code
This are just snipptes that were created while getting started with the OAK-D.
Most are copied examples from the depthai GitHub.

## 01 preview all cams
This example displays the preview image for all three cameras of the OAK-D at the same time.

## 02 tripple mobilenet
Inferencing with the mobilenet ssd is ran on all three cameras of the OAK-D at the same time.
The framerate drops a little.
All three networks run in parallel.

## 03 count persons
Again, three mobilenets are running in parallel.
This time, there is not output image.
Instead, just the number of detected persons is counted for each camera.
By comparing the results, they can correct themselfs (in theory).

## 04 persons depth
The main camera is used to detect persons.
The distance over the whole bounding box is averaged as a relative number.
The number of persons is then logged to a mariadb database, running on a local Raspberry Pi.

## 05 from scratch
After previously always modifying an existing example, for this code everything is written by hand.
The code simply outputs a 500x500 preview of the main camera.

## 09 laser control
Used to test the control interface to the gimbal system.

## 15 test with images
System performance was tested with the images captured at the bee hives.

## 16 test with video
The captured video from the visit to the bee keepers was tested with this code.