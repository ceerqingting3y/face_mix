cmake_minimum_required(VERSION 2.8.12)
# Every project needs a name.  We call this the "examples" project.
project(examples)

SET(OpenCV_DIR /Users/yingyu/opencv-3.4.3/release)

add_subdirectory(../../dlib dlib_build)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(face_morph face_morph.cpp src/delaunay.cpp src/utils.cpp)
target_link_libraries(face_morph dlib::dlib ${OpenCV_LIBS})

add_executable(face_morph_ex face_morph_ex.cpp src/utils.cpp)
target_link_libraries(face_morph_ex dlib::dlib ${OpenCV_LIBS} )

add_executable(static_transfer static_transfer.cpp src/delaunay.cpp src/utils.cpp src/dwt.cpp)
target_link_libraries(static_transfer dlib::dlib ${OpenCV_LIBS} )

add_executable(morph_video morph_video.cpp src/utils.cpp)
target_link_libraries(morph_video dlib::dlib ${OpenCV_LIBS} )

add_executable(generate_standard_face generate_standard_face.cpp src/delaunay.cpp src/utils.cpp)
target_link_libraries(generate_standard_face dlib::dlib ${OpenCV_LIBS})

add_executable(warp_to_happy warp_to_happy.cpp src/delaunay.cpp src/utils.cpp)
target_link_libraries(warp_to_happy dlib::dlib ${OpenCV_LIBS})

add_executable(wavelet_sample wavelet_sample.cpp src/wavelet.cpp)
target_link_libraries(wavelet_sample dlib::dlib ${OpenCV_LIBS})
