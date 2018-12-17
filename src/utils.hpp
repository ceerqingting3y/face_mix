#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/opencv.h>
using namespace dlib;
using namespace std;
std::vector<cv::Point2f> vectorize_landmarks(full_object_detection landmarks);
std::vector<cv::Point> int_vectorize_landmarks(full_object_detection landmarks);
cv::Point2f center_of_points(std::vector<cv::Point2f> points);
int clamp(int v, int a, int b);