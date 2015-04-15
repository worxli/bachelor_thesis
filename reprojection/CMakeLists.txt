cmake_minimum_required(VERSION 2.8)
project(calibrate)
set(
    CMAKE_RUNTIME_OUTPUT_DIRECTORY
    ${CMAKE_HOME_DIRECTORY}
    )
add_definitions(
    -Wfatal-errors
    )
add_executable(
    ${PROJECT_NAME}
    calibration.cpp
    )
target_link_libraries(
    ${PROJECT_NAME}
    -lopencv_calib3d
    -lopencv_core
    -lopencv_features2d
    -lopencv_highgui
    -lopencv_imgproc
    )

find_package( OpenCV REQUIRED )

project(calibrate2)
add_executable(calibrate2 calibration2.cpp)
find_package( OpenCV REQUIRED )
target_link_libraries( calibrate2 ${OpenCV_LIBS} )

project(stereo)
add_executable(stereo stereo_calib.cpp)
find_package( OpenCV REQUIRED )
target_link_libraries( stereo ${OpenCV_LIBS} )

project(reader)
add_executable(reader reader.cpp)
find_package( OpenCV REQUIRED )
target_link_libraries( reader ${OpenCV_LIBS} )