cmake_minimum_required(VERSION 3.2)

# Add depthai-core dependency
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/depthai-core EXCLUDE_FROM_ALL)

# Create a project with name 'myapp'
set(TARGET_NAME my-depthai-core-examples)
project(${TARGET_NAME})

# Dependencies (optional, only used for example)
find_package(OpenCV REQUIRED)

find_package(FP16 REQUIRED)
add_library(utility src/utility.cpp)
target_link_libraries(utility FP16::fp16 ${OpenCV_LIBS})

# Add source files
add_executable("gen2_camera_face_detector_example" 
    src/gen2_camera_face_detector_example.cpp
)

add_executable("gen2_movenet_example" 
    src/gen2_movenet_example.cpp
)

# Link with libraries
target_link_libraries("gen2_camera_face_detector_example" 
    PUBLIC 
        depthai-core
        utility
        ${OpenCV_LIBS} # optional, used for example
)

target_link_libraries("gen2_movenet_example" 
    PUBLIC 
        depthai-core
        utility
        depthai::opencv
        ${OpenCV_LIBS} # optional, used for example
)

# Set some errors
if(NOT MSVC)
    target_compile_options("gen2_camera_face_detector_example" PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Werror=return-type>)
    target_compile_options("gen2_movenet_example" PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Werror=return-type>)
endif()

# Set compiler features (c++11), and disables extensions (g++)
set_property(TARGET "gen2_camera_face_detector_example" PROPERTY CXX_STANDARD 11)
set_property(TARGET "gen2_camera_face_detector_example" PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET "gen2_camera_face_detector_example" PROPERTY CXX_EXTENSIONS OFF)

set_property(TARGET "gen2_movenet_example" PROPERTY CXX_STANDARD 11)
set_property(TARGET "gen2_movenet_example" PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET "gen2_movenet_example" PROPERTY CXX_EXTENSIONS OFF)