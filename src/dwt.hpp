#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include<iostream>
#include<math.h>

using namespace cv;
using namespace std;
void cal_dwt(cv::Mat & src, cv::Mat & info, cv::Mat & dst_show);
void cal_idwt(cv::Mat & src, cv::Mat & info, cv::Mat & dst_show);
void normalization(Mat & src, Mat & norm_src);

