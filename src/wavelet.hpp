#include <stdio.h>
#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

static void cvHaarWavelet(Mat &src, Mat &dst, int NIter);