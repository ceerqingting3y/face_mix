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
// Read points stored in the text files
std::vector<Point2f> write_points(string filename,
                                  std::vector<cv::Point2f> points) {
  ofstream ofs(filename);
  for (int i = 0; i < points.size(); i++) {
    ofs << points[i].x << " " << points[i].y << endl;
  }

  return points;
}

static cv::Rect dlib_rect_to_cv(dlib::rectangle r) {
  return cv::Rect(cv::Point2i(r.left(), r.top()),
                  cv::Point2i(r.right() + 1, r.bottom() + 1));
}

int main(int argc, char **argv) {
  string filename1("w_happy.jpg");

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

  // find a face and reformats it to the standard size of (80,80)
  std::vector<dlib::rectangle> faces;
  faces = detector(cimg);
  Mat cropped = src(dlib_rect_to_cv(faces[0]));
  Mat scaled;
  Size img_size(80, 80);
  resize(cropped, scaled, img_size);
  imshow("scaled", scaled);

  // update dlib variables
  cimg = cv_image<bgr_pixel>(scaled);
  faces = detector(cimg);

  image_window win;
  face_landmarks = pose_model(cimg, faces[0]);
  auto landmarks = vectorize_landmarks(face_landmarks);
  write_points("avg_face", landmarks);

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
