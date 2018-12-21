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
#include "src/delaunay.hpp"
#include "src/utils.hpp"
using namespace dlib;
using namespace std;

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
      std::vector<dlib::rectangle> faces = detector(cimg);
      // Find the pose of each face.
      full_object_detection face_landmarks;
      if (faces.size() > 0) {
        face_landmarks = pose_model(cimg, faces[0]);
        auto landmarks = vectorize_landmarks(face_landmarks);
        auto delaunay = get_delaunay(landmarks, rect);
        draw_delaunay(img, delaunay);
        // Display it all on the screen
        win.clear_overlay();
        win.set_image(cimg);
        win.add_overlay(render_face_detections(face_landmarks));
        cv::waitKey(30);
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
