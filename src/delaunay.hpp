#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void draw_delaunay(cv::Mat& img, cv::Subdiv2D& subdiv);

cv::Subdiv2D get_delaunay(std::vector<cv::Point2f>& landmarks, cv::Rect rect);

std::vector<std::vector<cv::Point>> vectorize_delaunay(cv::Subdiv2D& subdiv);

int landmark_closest_to_point(std::vector<cv::Point2f>& landmarks,
                              cv::Point& point);
std::vector<std::vector<int>> triangle_to_landmarks(
    std::vector<cv::Point2f>& landmarks, cv::Subdiv2D& subdiv);