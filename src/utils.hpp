#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/opencv.h>
using namespace dlib;
using namespace std;
std::vector<cv::Point2f> vectorize_landmarks(full_object_detection landmarks);