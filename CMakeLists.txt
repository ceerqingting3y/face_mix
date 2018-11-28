cmake_minimum_required(VERSION 2.8.12)
# Every project needs a name.  We call this the "examples" project.
project(examples)
add_subdirectory(../../dlib dlib_build)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(face_morph face_morph.cpp src/delaunay.cpp src/utils.cpp)
target_link_libraries(face_morph dlib::dlib ${OpenCV_LIBS})

add_executable(face_morph_ex face_morph_ex.cpp src/utils.cpp)
target_link_libraries(face_morph_ex dlib::dlib ${OpenCV_LIBS} )
