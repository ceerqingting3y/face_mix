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
  Mat wag = imread("wagner.jpg");
  Mat wagGray;
  cvtColor(wag, wagGray, COLOR_RGB2GRAY);
  wagGray.convertTo(wagGray, CV_32FC1);
  imshow("wag", wagGray);
  Mat transform = Mat::zeros(wagGray.size() * 2, CV_32FC1);
  int iter = 1;
  while (iter) {
    cvHaarWavelet(wagGray, transform, iter);
    imshow("Transform", transform);
    waitKey(0);
    iter++;
  }
}