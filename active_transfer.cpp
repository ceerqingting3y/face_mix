#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include "src/delaunay.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdlib.h>
#include <numeric>
#include <algorithm>
#include "src/dwt.hpp"
#include "src/hist.hpp"
#include "src/utils.hpp"
#include "src/transfer.hpp"

using namespace dlib;
using namespace std;
using namespace cv;

int main(int argc, char** argv){
	// Load face detection and pose estimation models.
    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor pose_model;
    deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

	// read target neutral triangle information 
	std::vector<std::vector<int>> triangles = read_triangles("jpg/baseline.txt");


	//------------------------------------------------------------------------------------------------------
	//read images 
	//------------------------------------------------------------------------------------------------------
	// read target image t (neutral)
	string filename_t = argv[2];
	cv::Mat target0 = cv::imread(filename_t);
	cv_image<bgr_pixel> img_t(target0);
	cv::Mat target;
	target0.convertTo(target, CV_32F);

	// Detect target
    std::vector<dlib::rectangle> tfaces = detector(img_t);
	full_object_detection tface_landmarks;
	tface_landmarks = pose_model(img_t, tfaces[0]);
	auto tpoints = vectorize_landmarks(tface_landmarks);
	Rect rect_face_t = boundingRect(tpoints);
    // save only the face part 
	Mat face_t;
	target0(rect_face_t).copyTo(face_t);
	
	// read source image s (non-neutral)
	string filename_s = argv[1];
	array2d<bgr_pixel> img_s;
	load_image(img_s, filename_s);
	cv::Mat source0 = dlib::toMat(img_s);
	Mat source;
	source0.convertTo(source,CV_32F);

    // detect source 
	std::vector<dlib::rectangle> sfaces = detector(img_s);
	full_object_detection sface_landmarks;
	sface_landmarks = pose_model(img_s, sfaces[0]);
	auto spoints = vectorize_landmarks(sface_landmarks);
	Rect rect_face_s = boundingRect(spoints);
    //save only the face part
	Mat face_s;
	source0(rect_face_s).copyTo(face_s);
	
	// try s1 to t1 transformation (show image )
	face_t.convertTo(face_t,CV_32FC3);
	face_s.convertTo(face_s,CV_32FC3);
	
    // face_t2 transfer face_t to face_s (get the emotional change of face_t)
	Mat face_t2 = Mat::zeros(face_s.rows,face_s.cols, CV_32FC3);

	for (std::vector<std::vector<int>>::iterator it = triangles.begin(); it!=triangles.end();++it){
		std::vector<Point2f> sr, tr;
		int a,b,c;
		a = (*it)[0];
		b = (*it)[1];
		c = (*it)[2];
		sr.push_back(spoints[a]);
		sr.push_back(spoints[b]);
		sr.push_back(spoints[c]);
		
		tr.push_back(tpoints[a]);
		tr.push_back(tpoints[b]);
		tr.push_back(tpoints[c]);
		
		Rect rs1 = boundingRect(sr);
		Rect rt1 = boundingRect(tr);
		
		std::vector<Point2f> triangle1,triangle2;
		//turn the points to the coordinates in the cropped face image 
		for (int i = 0; i < 3; i++) {
            //source image points
			triangle1.push_back(Point2f(sr[i].x - rect_face_s.x, sr[i].y - rect_face_s.y));
            //target image points 
			triangle2.push_back(Point2f(tr[i].x - rect_face_t.x, tr[i].y - rect_face_t.y));
		}

		transfer_to_source(triangle2,triangle1,face_t,face_t2);
    }
	Mat t2;
	face_t2.convertTo(t2, CV_8UC3);
	imshow("Morphed Face2", t2);
	//display it all on the screen
	waitKey(0);
	
    //motify mouth 

    // warp image back to its face 
}