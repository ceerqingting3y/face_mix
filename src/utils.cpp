#include "utils.hpp"
std::vector<cv::Point2f> vectorize_landmarks(full_object_detection landmarks) {
  std::vector<cv::Point2f> tmp;
  for (int i = 0; i < landmarks.num_parts(); i++) {
    point landmark = landmarks.part(i);
    tmp.push_back(cv::Point2f(landmark.x(), landmark.y()));
  }
  return tmp;
}

std::vector<cv::Point> int_vectorize_landmarks(
    full_object_detection landmarks) {
  std::vector<cv::Point> tmp;
  for (int i = 0; i < landmarks.num_parts(); i++) {
    point landmark = landmarks.part(i);
    tmp.push_back(cv::Point(landmark.x(), landmark.y()));
  }
  return tmp;
}
cv::Point2f center_of_points(std::vector<cv::Point2f> points) {
  float avgx(0), avgy(0);
  for (int i = 0; i < points.size(); i++) {
    avgx += points[i].x;
    avgy += points[i].y;
  }
  avgx /= points.size();
  avgy /= points.size();
  return cv::Point2f(avgx, avgy);
}
