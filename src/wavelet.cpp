#include <stdio.h>
#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

// from Andrey Smorodov
// disponible at
// https://stackoverflow.com/questions/20071854/wavelet-transform-in-opencv
//--------------------------------
// Wavelet transform
//--------------------------------
void cvHaarWavelet(Mat &src, Mat &dst, int NIter) {
  float c, dh, dv, dd;
  assert(src.type() == CV_32FC1);
  assert(dst.type() == CV_32FC1);
  int width = src.cols;
  int height = src.rows;
  for (int k = 0; k < NIter; k++) {
    for (int y = 0; y < (height >> (k + 1)); y++) {
      for (int x = 0; x < (width >> (k + 1)); x++) {
        c = (src.at<float>(2 * y, 2 * x) + src.at<float>(2 * y, 2 * x + 1) +
             src.at<float>(2 * y + 1, 2 * x) +
             src.at<float>(2 * y + 1, 2 * x + 1)) *
            0.5;
        dst.at<float>(y, x) = c;

        dh = (src.at<float>(2 * y, 2 * x) + src.at<float>(2 * y + 1, 2 * x) -
              src.at<float>(2 * y, 2 * x + 1) -
              src.at<float>(2 * y + 1, 2 * x + 1)) *
             0.5;
        dst.at<float>(y, x + (width >> (k + 1))) = dh;

        dv = (src.at<float>(2 * y, 2 * x) + src.at<float>(2 * y, 2 * x + 1) -
              src.at<float>(2 * y + 1, 2 * x) -
              src.at<float>(2 * y + 1, 2 * x + 1)) *
             0.5;
        dst.at<float>(y + (height >> (k + 1)), x) = dv;

        dd = (src.at<float>(2 * y, 2 * x) - src.at<float>(2 * y, 2 * x + 1) -
              src.at<float>(2 * y + 1, 2 * x) +
              src.at<float>(2 * y + 1, 2 * x + 1)) *
             0.5;
        dst.at<float>(y + (height >> (k + 1)), x + (width >> (k + 1))) = dd;
      }
    }
    dst.copyTo(src);
  }
}
