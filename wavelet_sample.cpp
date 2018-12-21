#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "src/delaunay.hpp"
#include "src/utils.hpp"
#include "src/wavelet.hpp"
using namespace dlib;
using namespace cv;
using namespace std;

int main() {
  Mat wag = imread("photos/monalisa.jpg");
  Mat wagGray;
  cvtColor(wag, wagGray, COLOR_RGB2GRAY);
  wagGray.convertTo(wagGray, CV_32FC1);
  Mat transform = Mat::zeros(wagGray.size(), CV_32FC1);
  cvHaarWavelet(wagGray, transform, 1);
  imshow("Transform", transform / 255);
  waitKey(0);
}