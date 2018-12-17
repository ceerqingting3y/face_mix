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
#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "src/delaunay.hpp"
#include "src/utils.hpp"
using namespace dlib;
using namespace cv;
using namespace std;

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
void morphFaces(Mat &src, Mat &base, Mat &output, Mat &allMask,
                std::vector<Point2f> &points1, std::vector<Point2f> &points2,
                double alpha) {
  // Find bounding rectangle for each triangle

  // Choose equivalent triangles in each image
  ifstream ifs("delaunay.txt");
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

    Rect r1 = boundingRect(t1);
    Rect r2 = boundingRect(t2);

    // Offset points by left top corner of the respective rectangles
    std::vector<Point2f> triangle1, triangle2;
    std::vector<Point> triangle2Int;
    for (int i = 0; i < 3; i++) {
      triangle1.push_back(Point2f(t1[i].x - r1.x, t1[i].y - r1.y));
      triangle2.push_back(Point2f(t2[i].x - r2.x, t2[i].y - r2.y));
      triangle2Int.push_back(
          Point(t2[i].x - r2.x, t2[i].y - r2.y));  // for fillConvexPoly
    }

    // Get mask by filling triangle
    Mat mask = Mat::zeros(r2.height, r2.width, CV_32FC3);
    fillConvexPoly(mask, triangle2Int, Scalar(1.0, 1.0, 1.0), 16, 0);
    // mask.convertTo(mask, CV_8UC1, 256);
    // Apply warpImage to small rectangular patches
    Mat srcRect, baseRect;
    src(r1).copyTo(srcRect);
    base(r2).copyTo(baseRect);

    Mat warpImage1 = Mat::zeros(r2.height, r2.width, srcRect.type());
    Mat warpImage2 = Mat::zeros(r2.height, r2.width, baseRect.type());

    applyAffineTransform(warpImage1, srcRect, triangle1, triangle2);
    warpImage2 = base(r2);
    // Alpha blend rectangular patches
    Mat imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2;

    // Copy triangular region of the rectangular patch to the output image

    // base(r2) = output;
    multiply(imgRect, mask, imgRect);
    multiply(output(r2), Scalar(1.0, 1.0, 1.0) - mask, output(r2));
    output(r2) = output(r2) + imgRect;
    allMask(r2) += allMask(r2) + mask;
  }
}

int main() {
  try {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
      cerr << "Unable to connect to camera" << endl;
      return 1;
    }

    image_window win;

    // Load face detection and pose estimation models.
    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor pose_model;
    deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

    // Grab and process frames until the\main window is closed by the user.
    while (!win.is_closed()) {
      // Grab a frame
      Mat src;
      if (!cap.read(src)) {
        break;
      }
      cv_image<bgr_pixel> cimg(src);
      Size size = src.size();
      Rect rect(0, 0, size.width, size.height);

      // Detect faces
      std::vector<dlib::rectangle> faces = detector(cimg);
      // Find the pose of each face.
      full_object_detection face_landmarks;
      if (faces.size() > 0) {
        face_landmarks = pose_model(cimg, faces[0]);
        auto landmarks = vectorize_landmarks(face_landmarks);

        // Read input images
        string filename2("ted_cruz.jpg");
        Mat base = imread(filename2);
        // convert Mat to float data type
        src.convertTo(src, CV_32F);
        base.convertTo(base, CV_32F);
        Mat output = base.clone();
        Mat mask = Mat::zeros(base.size(), CV_32FC3);

        double alpha = 0.1;

        // empty average image
        Mat imgMorph = Mat::zeros(src.size(), CV_32FC3);

        // Read points
        std::vector<Point2f> points1 = landmarks;
        std::vector<Point2f> points2 = readPoints(filename2 + ".txt");
        morphFaces(src, base, output, mask, points1, points2, alpha);
        Mat m1, m2;
        base.convertTo(m2, CV_8UC3);

        Point2f center(center_of_points(points2));
        Point centerInt((int)center.x, (int)center.y);
        mask.convertTo(mask, CV_8UC1, 256);
        output.convertTo(output, CV_8UC3);
        seamlessClone(output, m2, mask, centerInt, output, NORMAL_CLONE);
        imshow("Morphed Face", output);
        // Display it all on the screen
        waitKey(30);
      }
    }
  } catch (serialization_error &e) {
    cout << "You need dlib's default face landmarking model file to run this "
            "example."
         << endl;
    cout << "You can get it from the following URL: " << endl;
    cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
         << endl;
    cout << endl << e.what() << endl;
  } catch (exception &e) {
    cout << e.what() << endl;
  }
}
