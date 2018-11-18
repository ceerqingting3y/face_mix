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

using namespace dlib;
using namespace std;

std::vector<cv::Point2f> landmarkToCV(full_object_detection landmarks) {
  std::vector<cv::Point2f> tmp;
  for (int i = 0; i < landmarks.num_parts(); i++) {
    point landmark = landmarks.part(i);
    tmp.push_back(cv::Point2f(landmark.x(), landmark.y()));
  }
  cout << "tmp size : " << tmp.size() << endl;
  return tmp;
}

static void draw_delaunay(cv::Mat& img, cv::Subdiv2D& subdiv) {
  cv::Scalar delaunay_color(255, 255, 255);
  std::vector<cv::Vec6f> triangleList;
  subdiv.getTriangleList(triangleList);
  std::vector<cv::Point> pt(3);
  cv::Size size = img.size();
  cv::Rect rect(0, 0, size.width, size.height);

  for (size_t i = 0; i < triangleList.size(); i++) {
    cv::Vec6f t = triangleList[i];
    pt[0] = cv::Point(cvRound(t[0]), cvRound(t[1]));
    pt[1] = cv::Point(cvRound(t[2]), cvRound(t[3]));
    pt[2] = cv::Point(cvRound(t[4]), cvRound(t[5]));

    // Draw rectangles completely inside the image.
    if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2])) {
      cv::line(img, pt[0], pt[1], delaunay_color, 1, CV_AA, 0);
      cv::line(img, pt[1], pt[2], delaunay_color, 1, CV_AA, 0);
      cv::line(img, pt[2], pt[0], delaunay_color, 1, CV_AA, 0);
    }
  }
}

cv::Subdiv2D get_delaunay(std::vector<cv::Point2f>& landmarks, cv::Rect rect) {
  cv::Subdiv2D delaunay(rect);
  for (int i = 0; i < landmarks.size(); i++) {
    delaunay.insert(landmarks[i]);
  }
  return delaunay;
}

int main() {
  try {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
      cerr << "Unable to connect to camera" << endl;
      return 1;
    }

    image_window win;

    // Load face detection and pose estimation models.
    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor pose_model;
    deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

    // Grab and process frames until the main window is closed by the user.
    while (!win.is_closed()) {
      // Grab a frame
      cv::Mat img;
      if (!cap.read(img)) {
        break;
      }
      cv_image<bgr_pixel> cimg(img);
      cv::Size size = img.size();
      cv::Rect rect(0, 0, size.width, size.height);

      // Detect faces
      std::vector<rectangle> faces = detector(cimg);
      // Find the pose of each face.
      full_object_detection face_landmarks;
      if (faces.size() > 0) {
        face_landmarks = pose_model(cimg, faces[0]);
        auto landmarks = landmarkToCV(face_landmarks);
        auto delaunay = get_delaunay(landmarks, rect);
        draw_delaunay(img, delaunay);
        // Display it all on the screen
        win.clear_overlay();
        win.set_image(cimg);
        win.add_overlay(render_face_detections(face_landmarks));
        cv::waitKey(1000);
      }
    }
  } catch (serialization_error& e) {
    cout << "You need dlib's default face landmarking model file to run this "
            "example."
         << endl;
    cout << "You can get it from the following URL: " << endl;
    cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
         << endl;
    cout << endl << e.what() << endl;
  } catch (exception& e) {
    cout << e.what() << endl;
  }
}
