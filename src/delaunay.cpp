#include "delaunay.hpp"

void draw_delaunay(cv::Mat& img, cv::Subdiv2D& subdiv) {
  auto triangles = vectorize_delaunay(subdiv);
  cv::Scalar delaunay_color(255, 255, 255);
  cv::Size size = img.size();
  cv::Rect rect(0, 0, size.width, size.height);
  for (size_t i = 0; i < triangles.size(); i++) {
    auto pt = triangles[i];
    // Draw rectangles completely inside the image.
    if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2])) {
      cv::line(img, pt[0], pt[1], delaunay_color, 1, CV_AA, 0);
      cv::line(img, pt[1], pt[2], delaunay_color, 1, CV_AA, 0);
      cv::line(img, pt[2], pt[0], delaunay_color, 1, CV_AA, 0);
    }
  }
}

std::vector<std::vector<cv::Point>> vectorize_delaunay(cv::Subdiv2D& subdiv) {
  std::vector<cv::Vec6f> triangleList;
  subdiv.getTriangleList(triangleList);
  std::vector<cv::Point> pt(3);
  std::vector<std::vector<cv::Point>> list_of_pt;
  for (size_t i = 0; i < triangleList.size(); i++) {
    cv::Vec6f t = triangleList[i];
    pt[0] = cv::Point(cvRound(t[0]), cvRound(t[1]));
    pt[1] = cv::Point(cvRound(t[2]), cvRound(t[3]));
    pt[2] = cv::Point(cvRound(t[4]), cvRound(t[5]));
    list_of_pt.push_back(pt);
  }
  return list_of_pt;
}

int landmark_closest_to_point(std::vector<cv::Point2f>& landmarks,
                              cv::Point& point) {
  int max = 1e9;
  int landmark = 0;
  cv::Point2f aux(point);
  for (int i = 0; i < landmarks.size(); i++) {
    double norm = cv::norm(landmarks[i] - aux);
    if (norm < max) {
      max = norm;
      landmark = i;
    }
  }
  return landmark;
}

std::vector<std::vector<int>> triangle_to_landmarks(
    std::vector<cv::Point2f>& landmarks, cv::Subdiv2D& subdiv) {
  std::vector<std::vector<int>> list_of_landmarks;
  auto triangles = vectorize_delaunay(subdiv);
  for (int i = 0; i < triangles.size(); i++) {
    list_of_landmarks.push_back(std::vector<int>());
    for (int j = 0; j < 3; j++) {
      int land = landmark_closest_to_point(landmarks, triangles[i][j]);
      list_of_landmarks[i].push_back(land);
    }
  }
  return list_of_landmarks;
}

cv::Subdiv2D get_delaunay(std::vector<cv::Point2f>& landmarks, cv::Rect rect) {
  cv::Subdiv2D delaunay(rect);
  for (int i = 0; i < landmarks.size(); i++) {
    delaunay.insert(landmarks[i]);
  }
  return delaunay;
}