<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <!-- <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul> -->
    </li>
    <li>
      <a href="#Prerequisites">Prerequisites</a>
      <!-- <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul> -->
    </li>
    <li><a href="#Building-AVP-library">Building AVP library</a></li>
    <li><a href="#Testing">Testing</a></li>
    <!-- <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li> -->
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
This project aims to address the problem of mapping an underwater structure by a team of co-robots. Our approach includes underwater state estimation, 3D motion planning underwater and limited bandwidth communications
We use stereo camera and sonar as the primary sensors. 


<!-- ### Built With

This section should list any major frameworks that you built your project using. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples.
* [Bootstrap](https://getbootstrap.com)
* [JQuery](https://jquery.com)
* [Laravel](https://laravel.com) -->



<!-- GETTING STARTED -->
<!-- ## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps. -->

### Prerequisites
We have tested the library in Ubuntu 16.04 and 18.04, but it should be easy to compile in other platforms. 
This is an example of how to list things you need to use the software and how to install them.
<!-- * npm
  ```sh
  npm install npm@latest -g
  ``` -->
#### C++11 or C++0x Compiler
We use the new thread and chrono functionalities of C++11.

####  CMAKE
```
sudo apt install cmake
```

####  OpenCV
We use [OpenCV](http://opencv.org) to manipulate images and features. Download and install instructions can be found at: http://opencv.org. **Required at leat 2.4.3. Tested with OpenCV 2.4.11 and OpenCV 3.2**.


####  pybind11
We use [pybind11](https://pybind11.readthedocs.io/en/stable/compiling.html) to creat Python bindings of existing c++ code.  Download and install instructions can be found at:: https://github.com/pybind/pybind11.

####  Eigen
```
sudo apt install libeigen3-dev
```
####  Anaconda
We highly recommend installing a [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/distribution/#download-section) environment (note: python>=3.6 is required). Once you have Anaconda installed, here are the instructions.



### Building AVP library
1. Clone the repository:
  ```
  git clone https://github.com/wwtx9/AVP_Binding.git AVP_Binding
  ```
2. Build and compire AVP library
  ```
  cd AVP_Binding
  mkdir build
  cmake ..
  make 
  ```
3. Setup python virtual environment with conda
```
# We require python>=3.6 and cmake>=3.10
   conda create -n AVP python=3.6 cmake=3.14.0
   conda activate AVP
```
4. Install the AVP Library for your python project to import
  ```
  pip install .
  #After that, now your can import AVP_Binding as m in your python script
  ```

### Testing
#### KITTI Dataset  
1. Download the dataset (grayscale images) from http://www.cvlibs.net/datasets/kitti/eval_odometry.php 

2. Run test script
```
# Assuming we're still within AVP conda environment
    python test.py
```

