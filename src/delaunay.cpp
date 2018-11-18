#include "delaunay.hpp"

void draw_delaunay(cv::Mat& img, cv::Subdiv2D& subdiv) {
  cv::Scalar delaunay_color(255, 255, 255);
  std::vector<cv::Vec6f> triangleList;
  subdiv.getTriangleList(triangleList);
  std::vector<cv::Point> pt(3);
  cv::Size size = img.size();
  cv::Rect rect(0, 0, size.width, size.height);

  for (size_t i = 0; i < triangleList.size(); i++) {
    cv::Vec6f t = triangleList[i];
    pt[0] = cv::Point(cvRound(t[0]), cvRound(t[1]));
    pt[1] = cv::Point(cvRound(t[2]), cvRound(t[3]));
    pt[2] = cv::Point(cvRound(t[4]), cvRound(t[5]));

    // Draw rectangles completely inside the image.
    if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2])) {
      cv::line(img, pt[0], pt[1], delaunay_color, 1, CV_AA, 0);
      cv::line(img, pt[1], pt[2], delaunay_color, 1, CV_AA, 0);
      cv::line(img, pt[2], pt[0], delaunay_color, 1, CV_AA, 0);
    }
  }
}

cv::Subdiv2D get_delaunay(std::vector<cv::Point2f>& landmarks, cv::Rect rect) {
  cv::Subdiv2D delaunay(rect);
  for (int i = 0; i < landmarks.size(); i++) {
    delaunay.insert(landmarks[i]);
  }
  return delaunay;
}