cmake_minimum_required(VERSION 2.8.12)
# Every project needs a name.  We call this the "examples" project.
project(examples)

SET(OpenCV_DIR /Users/yingyu/opencv-3.4.3/release)

add_subdirectory(../../dlib dlib_build)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(face_morph face_morph.cpp src/delaunay.cpp src/utils.cpp)
target_link_libraries(face_morph dlib::dlib ${OpenCV_LIBS})
set_target_properties(face_morph PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_executable(face_morph_ex face_morph_ex.cpp src/utils.cpp)
target_link_libraries(face_morph_ex dlib::dlib ${OpenCV_LIBS} )
set_target_properties(face_morph_ex PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_executable(static_transfer static_transfer.cpp src/delaunay.cpp src/utils.cpp src/dwt.cpp src/hist.cpp src/transfer.cpp)
target_link_libraries(static_transfer dlib::dlib ${OpenCV_LIBS} )
set_target_properties(static_transfer PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

<<<<<<< HEAD
add_executable(active_transfer active_transfer.cpp src/delaunay.cpp src/utils.cpp src/dwt.cpp src/hist.cpp src/transfer.cpp)
target_link_libraries(active_transfer dlib::dlib ${OpenCV_LIBS} )
=======
>>>>>>> 9f96e0939d6c04689b85955cad52e54a8bf8eabd

add_executable(morph_video morph_video.cpp src/utils.cpp)
target_link_libraries(morph_video dlib::dlib ${OpenCV_LIBS} )
set_target_properties(morph_video PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")


add_executable(generate_standard_face generate_standard_face.cpp src/delaunay.cpp src/utils.cpp)
target_link_libraries(generate_standard_face dlib::dlib ${OpenCV_LIBS})
set_target_properties(generate_standard_face PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")



add_executable(warp_to_happy warp_to_happy.cpp src/delaunay.cpp src/utils.cpp)
target_link_libraries(warp_to_happy dlib::dlib ${OpenCV_LIBS})
set_target_properties(warp_to_happy PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")



add_executable(wavelet_sample wavelet_sample.cpp src/wavelet.cpp)
target_link_libraries(wavelet_sample dlib::dlib ${OpenCV_LIBS})
<<<<<<< HEAD
=======
set_target_properties(wavelet_sample PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")


add_executable(write_delaunay write_delaunay.cpp src/delaunay.cpp src/utils.cpp)
target_link_libraries(write_delaunay dlib::dlib ${OpenCV_LIBS})
set_target_properties(write_delaunay PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

>>>>>>> 9f96e0939d6c04689b85955cad52e54a8bf8eabd
