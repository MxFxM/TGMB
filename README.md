# TGMB
The German Mite Busters Repository for the OpenCV AI Competition 2021

![detection results](https://raw.githubusercontent.com/MxFxM/TGMB/main/detection_comparison/20210530150512_det_4.png "Bee detection results")

# Documentation
Documentation is here and a short video seriese on YouTube:

https://www.youtube.com/playlist?list=PLd_aTu_oC2t4qxTNdH-ToK-ty6c6wK3zz

# Setup
Create a virtual environment: python3 -m venv tgmb-env

Source the environment: source tgmb-env/bin/activate

Install dependencies: python3 -m pip install -r requirements.txt

You might have to set the udev rules: echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules

And then reload: sudo udevadm control --reload-rules && sudo udevadm trigger

# Login Credentials
When using secret credentials for logging in to different services, add a file called "credentials.py" into the directory "code".
This file is ignored in the .gitignore.
You can import this file in your scripts.

# Requirements for code
04 requires credentials for mariadb database.
Also the python package mariadb (get with pip) is required.
The later code needs access to the serial port.
