# TGMB
The German Mite Busters Repository for the OpenCV AI Competition 2021

# Setup
Create a virtual environment: python3 -m venv tgmb-env

Source the environment: source tgmb-env/bin/activate

Install dependencies: python3 -m pip install -r requirements.txt

# Login Credentials
When using secret credentials for logging in to different services, add a file called "credentials.py" into the directory "code".
This file is ignored in the .gitignore.
You can import this file in your scripts.

# Requirements for code
04 requires credentials for mariadb database.
Also the python package mariadb (get with pip) is required.
