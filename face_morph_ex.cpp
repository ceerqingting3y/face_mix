
#include <stdlib.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// Read points stored in the text files
vector<Point2f> readPoints(string pointsFileName) {
  vector<Point2f> points;
  ifstream ifs(pointsFileName);
  float x, y;
  while (ifs >> x >> y) {
    points.push_back(Point2f(x, y));
  }

  return points;
}

// Apply affine transform calculated using srcTri and dstTri to src
void applyAffineTransform(Mat &warpImage, Mat &src, vector<Point2f> &srcTri,
                          vector<Point2f> &dstTri) {
  // Given a pair of triangles, find the affine transform.
  Mat warpMat = getAffineTransform(srcTri, dstTri);

  // Apply the Affine Transform just found to the src image
  warpAffine(src, warpImage, warpMat, warpImage.size(), INTER_LINEAR,
             BORDER_REFLECT_101);
}

// Warps and alpha blends triangular regions from img1 and img2 to img
void morphFaces(Mat &img1, Mat &img2, vector<Point2f> &t1, vector<Point2f> &t2,
                double alpha) {
  // Find bounding rectangle for each triangle

  Rect r1 = boundingRect(t1);
  Rect r2 = boundingRect(t2);

  // Offset points by left top corner of the respective rectangles
  vector<Point2f> triangle1, triangle2;
  vector<Point> triangle2Int;
  for (int i = 0; i < 3; i++) {
    triangle1.push_back(Point2f(t1[i].x - r1.x, t1[i].y - r1.y));
    triangle2.push_back(Point2f(t2[i].x - r2.x, t2[i].y - r2.y));
    triangle2Int.push_back(
        Point(t2[i].x - r2.x, t2[i].y - r2.y));  // for fillConvexPoly
  }

  // Get mask by filling triangle
  // Mat mask = Mat::ones(r2.height, r2.width, CV_8UC1) * 256;
  Mat mask = Mat::zeros(r2.height, r2.width, CV_32FC3);
  fillConvexPoly(mask, triangle2Int, Scalar(1.0, 1.0, 1.0), 16, 0);
  mask.convertTo(mask, CV_8UC1, 256);
  // Apply warpImage to small rectangular patches
  Mat img1Rect, img2Rect;
  img1(r1).copyTo(img1Rect);
  img2(r2).copyTo(img2Rect);

  Mat warpImage1 = Mat::zeros(r2.height, r2.width, img1Rect.type());
  Mat warpImage2 = Mat::zeros(r2.height, r2.width, img2Rect.type());

  applyAffineTransform(warpImage1, img1Rect, triangle1, triangle2);
  warpImage2 = img2(r2);
  // Alpha blend rectangular patches
  Mat imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2;

  // Copy triangular region of the rectangular patch to the output image
  Mat output;
  Point center = Point(r2.width / 2, r2.height / 2);
  Mat m1, m2;
  imgRect.convertTo(m1, CV_8UC3);
  img2(r2).convertTo(m2, CV_8UC3);
  seamlessClone(m1, m2, mask, center, output, NORMAL_CLONE);
  img2(r2) = output;
  // multiply(imgRect, mask, imgRect);
  // multiply(img2(r2), Scalar(1.0, 1.0, 1.0) - mask, img2(r2));
  // img2(r2) = img2(r2) + imgRect;
}

int main(int argc, char **argv) {
  string filename1("wagner.jpg");
  string filename2("mona.jpg");

  // alpha controls the degree of morph
  double alpha = 0.1;

  // Read input images
  Mat img1 = imread(filename1);
  Mat img2 = imread(filename2);

  // convert Mat to float data type
  img1.convertTo(img1, CV_32F);
  img2.convertTo(img2, CV_32F);

  // empty average image
  Mat imgMorph = Mat::zeros(img1.size(), CV_32FC3);

  // Read points
  vector<Point2f> points1 = readPoints(filename1 + ".txt");
  vector<Point2f> points2 = readPoints(filename2 + ".txt");
  vector<Point2f> points;

  // compute weighted average point coordinates
  for (int i = 0; i < points1.size(); i++) {
    float x, y;
    x = (1 - alpha) * points1[i].x + alpha * points2[i].x;
    y = (1 - alpha) * points1[i].y + alpha * points2[i].y;

    points.push_back(Point2f(x, y));
  }

  // Read triangle indices
  ifstream ifs("delaunay.txt");
  int x, y, z;

  vector<Point2f> t1, t2, t;
  while (ifs >> x >> y >> z) {
    // Triangles

    // Triangle corners for image 1.
    t1.push_back(points1[x]);
    t1.push_back(points1[y]);
    t1.push_back(points1[z]);

    // Triangle corners for image 2.
    t2.push_back(points2[x]);
    t2.push_back(points2[y]);
    t2.push_back(points2[z]);

    // Triangle corners for morphed image.
    t.push_back(points[x]);
    t.push_back(points[y]);
    t.push_back(points[z]);
    morphFaces(img1, img2, t1, t2, alpha);
  }

  // Display Result
  imshow("Morphed Face", img2 / 255.0);
  imwrite("wagnelisa.jpg", img2);
  waitKey(0);

  return 0;
}
