# SAILenv Client - SAILab Virtual Environment 

SAILenv is a Virtual Environment powered by Unity3D, developed by SAILab in Siena. This repository contains a Python client for SAILenv.

## Agent API

SAILenv is accessible through an instance of the Agent class, which acts as the API between Python scripts and the environment. The API is designed to be as easy as possible to integrate with most known computer vision frameworks.

Through the API, a Python script can access many views of the environment with configurable resolution scaling (aspect ratio is fixed, at the moment):

* Realistic rendering of the environment
* Pixel-wise segmentation at category level
* Pixel-wise segmentation at instance level
* Pixel-wise depth
* Pixel-wise optical flow

A script can also:
* Obtain the list of object categories available in the scene
* Obtain the list of available scenes
* Transfer the Agent in one of the available scenes
* Manually set the Agent position and orientation
* Enable automatic movement through a predetermined track inside the scene

### Optical Flow
The optical flow is obtained directly from the Unity Physics Engine, meaning that it is both highly accurate while still achieving greats speed.  

### Agent and Environment communication
The Agent and the Environment communicate through low-level sockets, allowing highly performant exchange of information. 


## Download
[![PyPI version](https://badge.fury.io/py/sailenv.svg)](https://badge.fury.io/py/sailenv)