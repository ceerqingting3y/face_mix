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

// Read points stored in the text files
void write_points(string filename, std::vector<cv::Point> points) {
  ofstream ofs(filename);
  for (int i = 0; i < points.size(); i++) {
    ofs << points[i].x << " " << points[i].y << endl;
  }
}

static cv::Rect dlib_rect_to_cv(dlib::rectangle r) {
  return cv::Rect(cv::Point2i(r.left(), r.top()),
                  cv::Point2i(r.right() + 1, r.bottom() + 1));
}

Mat genMask(std::vector<Point> &points) {
  // Find bounding rectangle for each triangle
  // Read triangle indices
  ifstream ifs("delaunay.txt");
  int x, y, z;
  Mat allMask = Mat::zeros(IMG_SIZE, IMG_SIZE, CV_8UC3);
  while (ifs >> x >> y >> z) {
    // Triangles
    std::vector<Point2f> t;

    // Triangle corners for image 1.
    t.push_back(points[x]);
    t.push_back(points[y]);
    t.push_back(points[z]);

    // Offset points by left top corner of the respective rectangles
    std::vector<Point> triangleInt;
    for (int i = 0; i < 3; i++) {
      triangleInt.push_back(Point(t[i].x, t[i].y));  // for fillConvexPoly
    }

    // Get mask by filling triangle
    Mat mask = Mat::zeros(IMG_SIZE, IMG_SIZE, CV_8UC3);
    fillConvexPoly(mask, triangleInt, Scalar(255, 255, 255), 16, 0);
    allMask = allMask + mask;
  }
  allMask.convertTo(allMask, CV_8UC3);
  return allMask;
}

int main(int argc, char **argv) {
  if (argc == 1) {
    cerr << "Please provide the photo name" << endl;
    return 1;
  }
  string filename1(argv[1]);

  // alpha controls the degree of morph
  double alpha = 1;
  // Read input images
  Mat src = imread(filename1);

  // initial setup for dlib
  cv_image<bgr_pixel> cimg(src);
  matrix<rgb_pixel> simg, cut_img;
  frontal_face_detector detector = get_frontal_face_detector();
  shape_predictor pose_model;
  deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

  full_object_detection face_landmarks;

  // find a face and reformats it to the standard size of (IMG_SIZE,IMG_SIZE)
  std::vector<dlib::rectangle> faces;
  faces = detector(cimg);
  Mat cropped = src(dlib_rect_to_cv(faces[0]));
  Mat scaled;
  Size img_size(IMG_SIZE, IMG_SIZE);
  resize(cropped, scaled, img_size);
  imshow("scaled", scaled);

  // update dlib variables
  cimg = cv_image<bgr_pixel>(scaled);
  faces = detector(cimg);

  image_window win;
  face_landmarks = pose_model(cimg, faces[0]);
  auto landmarks = int_vectorize_landmarks(face_landmarks);
  write_points("avg_face", landmarks);

  Mat mask = genMask(landmarks);
  Mat cropped2 = Mat::zeros(IMG_SIZE, IMG_SIZE, CV_8UC3);
  scaled.copyTo(cropped2, mask);
  imshow("cropped", cropped2);
  imwrite("wagner_lindo.jpg", cropped2);
  // Read points
  dlib::assign_image(simg, cimg);
  //   dlib::extract_image_chips(simg, faces[0], cut_img);

  win.clear_overlay();
  win.set_image(simg);
  win.add_overlay(faces, rgb_pixel(255, 0, 0));
  win.add_overlay(render_face_detections(face_landmarks));

  imshow("wagner", src);
  waitKey(0);

  return 0;
}
