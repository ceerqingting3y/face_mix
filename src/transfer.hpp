#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdlib.h>
#include <numeric>
#include <algorithm>

using namespace dlib;
using namespace std;
using namespace cv;

std::vector<std::vector<int>> read_triangles(string filename);
int search_point(std::vector<Point2f> &points, Point2f & point);
std::vector<std::vector<int>> get_triangles_from_target(cv::Mat &image, std::vector<cv::Point2f> &points, string filename);
void show_landmarks(cv::Mat image, std::vector<cv::Point2f> &points);
void show_delauney(cv::Mat image, std::vector<cv::Point2f> &points, std::vector<std::vector<int>> &triangles);
cv::Mat myGetAffineTransform(const std::vector<Point2f> & src, const std::vector<Point2f> &dst, int m);
void applyAffineTransform(Mat &warpImage, Mat &src, std::vector<Point2f> &srcTri, std::vector<Point2f> &dstTri);
void transfer_to_source(std::vector<Point2f> & src_tri,std::vector<Point2f> & dst_tri,  Mat & src_face, Mat &dst_face);
void target_to_target(std::vector<Point2f> & t1_tri,std::vector<Point2f> & s1_tri, std::vector<Point2f> & s2_tri, std::vector<Point> & t_int, Mat & face_t1,Mat &face_t2, Mat & face_s1, Mat & face_s2);
void calculate_new_points(std::vector<std::vector<Point2f>> & t2_points,std::vector<Point2f> &t2_new_points);
void change_face(std::vector<Point2f> & src_tri,std::vector<Point2f> & dst_tri,  Mat & src_face, Mat &dst_face, std::vector<Point2f> & src_3,std::vector<Point2f> & dst_3);
void similarityTransform(std::vector<cv::Point2f>& inPoints, std::vector<cv::Point2f>& outPoints, cv::Mat &tform);
void get_face_change_matrix(std::vector<cv::Point2f>& inPoints, std::vector<cv::Point2f>& outPoints, cv::Mat &tform);
