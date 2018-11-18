#include "utils.hpp"
std::vector<cv::Point2f> vectorize_landmarks(full_object_detection landmarks) {
  std::vector<cv::Point2f> tmp;
  for (int i = 0; i < landmarks.num_parts(); i++) {
    point landmark = landmarks.part(i);
    tmp.push_back(cv::Point2f(landmark.x(), landmark.y()));
  }
  return tmp;
}
