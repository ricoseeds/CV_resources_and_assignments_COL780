Repository for COL780

**External libs**

- Eigen3
- nlohmann/json


Enabling VIZ in OpenCV in Visual Studio

VIZ is a contrib module and requies special build steps in OpenCV
It is enabled only when OpenCV is buuld with VTK. Which consequently requires building VTK. It is possible that prebuild binaries are available on Linux or MacOS but for windows I had to compile.

1. Download VTK, the version I used is vtk-v8.2.0 available at: https://gitlab.kitware.com/vtk/vtk/-/tags
2. Build the VTK and keep the default cmake flags as it is, no need to change anything.
3. Also give the path to install VTK as below
CMAKE_INSTALL_PREFIX = C:/p/vtk-v8.2.0/install
4. After building go to INSTALL and build it, that will copy all the built binaries to the above directory. So now VTK build step is done.
5. Building OpenCV with VTK enable flag WITH_VTK
Also give VTK_DIR as C:/p/vtk-v8.2.0/install/lib/cmake/vtk-8.2

Further set OPENCV_EXTRA_MODULES_PATH as C:/Projects/opencv_contrib-4.1.1/modules, this is because VIZ like xfeatures2d does not come along with default opencv but comes inside contrib modules. If you dont have contrib you need to doenload that as well.
6. run CMAKE with above configuation, and you should see BUILD_opencv_viz as an option in the cmake gui, that will eb enavbled because we provided VTK.
7.Build OpenCV, thats all now you should be able to use viz module inside your project.


Camera Calibration Resources
-----------------------------

Calibration pattern
https://docs.opencv.org/master/da/d0d/tutorial_camera_calibration_pattern.html

OpenCV Tutorial
https://docs.opencv.org/4.1.1/d4/d94/tutorial_camera_calibration.html

Helpful videos
https://www.youtube.com/watch?v=HNfPbw-1e_w
https://www.youtube.com/watch?v=v7jutAmWJVQ
















