// The contents of this file are in the public domain. See
// LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example program shows how to find frontal human faces in an image and
    estimate their pose.  The pose takes the form of 68 landmarks.  These are
    points on the face such as the corners of the mouth, along the eyebrows, on
    the eyes, and so forth.


    This example is essentially just a version of the
   face_landmark_detection_ex.cpp example modified to use OpenCV's VideoCapture
   object to read from a camera instead of files.


    Finally, note that the face detector is fastest when compiled with at least
    SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
    chip then you should enable at least SSE2 instructions.  If you are using
    cmake to compile this program you can enable them by using one of the
    following commands when you create the build project:
        cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
    This will set the appropriate compiler options for GCC, clang, Visual
    Studio, or the Intel compiler.  If you are using another compiler then you
    need to consult your compiler's manual to determine how to enable these
    instructions.  Note that AVX is the fastest but requires a CPU from at least
    2011.  SSE4 is the next fastest and is supported by most current machines.
*/

#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_transforms.h>
#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "src/delaunay.hpp"
#include "src/utils.hpp"
using namespace dlib;
using namespace cv;
using namespace std;

const int IMG_SIZE = 160;

std::vector<Point2f> readPoints(string pointsFileName) {
  std::vector<Point2f> points;
  ifstream ifs(pointsFileName);
  float x, y;
  while (ifs >> x >> y) {
    points.push_back(Point2f(x, y));
  }

  return points;
}

// Apply affine transform calculated using srcTri and dstTri to src
void applyAffineTransform(Mat &warpImage, Mat &src,
                          std::vector<Point2f> &srcTri,
                          std::vector<Point2f> &dstTri) {
  // Given a pair of triangles, find the affine transform.
  Mat warpMat = getAffineTransform(srcTri, dstTri);

  // Apply the Affine Transform just found to the src image
  warpAffine(src, warpImage, warpMat, warpImage.size(), INTER_LINEAR,
             BORDER_REFLECT_101);
}

// Warps and alpha blends triangular regions from src and base to img
void warpFaces(Mat &src, Mat &output, std::vector<Point2f> &points1,
               std::vector<Point2f> &points2) {
  // Find bounding rectangle for each triangle

  // Choose equivalent triangles in each image
  ifstream ifs("jpg/baseline.txt");
  int x, y, z;
  while (ifs >> x >> y >> z) {
    // Triangles
    std::vector<Point2f> t1, t2;

    // Triangle corners for image 1.
    t1.push_back(points1[x]);
    t1.push_back(points1[y]);
    t1.push_back(points1[z]);

    // Triangle corners for image 2.
    t2.push_back(points2[x]);
    t2.push_back(points2[y]);
    t2.push_back(points2[z]);

    // Offset points by left top corner of the respective rectangles
    std::vector<Point2f> triangle1, triangle2;
    std::vector<Point> triangle2Int;
    for (int i = 0; i < 3; i++) {
      triangle1.push_back(Point2f(t1[i].x, t1[i].y));
      triangle2.push_back(Point2f(t2[i].x, t2[i].y));
      triangle2Int.push_back(Point(t2[i].x, t2[i].y));
    }

    // Get mask by filling triangle
    // mask.convertTo(mask, CV_8UC1, 256);
    // Apply warpImage to small rectangular patches
    Mat warpImage(IMG_SIZE, IMG_SIZE, CV_32FC3);
    Mat mask = Mat::zeros(IMG_SIZE, IMG_SIZE, CV_8UC3);
    fillConvexPoly(mask, triangle2Int, Scalar(255, 255, 255), 16, 0);
    mask.convertTo(mask, CV_8U);
    applyAffineTransform(warpImage, src, triangle1, triangle2);
    warpImage.copyTo(output, mask);

    // Copy triangular region of the rectangular patch to the output image

    // base(r2) = output;
  }
}

int main() {
  // Grab and process frames until the\main window is closed by the user.
  // Grab a frame
  Mat src;
  src = imread("photos/wagner_neutral.jpg");
  Mat shouting = imread("photos/wagner_lindo.jpg");
  // Detect faces

  // Read input images
  Mat output = Mat::zeros(src.size(), CV_32FC3);
  // convert Mat to float data type

  // Read points
  std::vector<Point2f> points1 = readPoints("avg_face_neutral.txt");
  std::vector<Point2f> points2 = readPoints("avg_face_shouting.txt");
  warpFaces(src, output, points1, points2);
  Mat difference;

  absdiff(shouting, output, difference);
  imshow("difference", difference);
  imshow("neutral", src);
  imshow("shouting", shouting);
  imshow("Morphed Face", output);
  // Display it all on the screen
  waitKey(0);
}