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
	//----------------------------------------------------------------------------------------------------------
	// Part I: read target image + get webcam input frame + detect landmarks
	//----------------------------------------------------------------------------------------------------------
	shape_predictor pose_model;
	deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
	// Load face detection and pose estimation models.
	frontal_face_detector detector = get_frontal_face_detector();
	// read target neutral triangle information 
	std::vector<std::vector<int>> triangles = read_triangles("jpg/baseline.txt");
	try {
		VideoCapture cap(0);
		if (!cap.isOpened()) {
		cerr << "Unable to connect to camera" << endl;
		return 1;
		}
		namedWindow("Change Expression");
		namedWindow("Real face");
	while (true) {
		// Grab a frame
		Mat source0;
		if (!cap.read(source0)) {
		break;
		}
		imshow("Real face",source0);
		//waitKey(30);

		// read target image t (neutral)
		string filename_t = argv[1];
		cv::Mat target0 = cv::imread(filename_t);
		cv_image<bgr_pixel> img_t(target0);
		cv::Mat target;
		target0.convertTo(target, CV_32FC3);

		// Detect target
		std::vector<dlib::rectangle> tfaces = detector(img_t);
		if(tfaces.size()==0){
			cout<<"There is no face in the image! Please change!"<<endl;
			return 1;
		}
		full_object_detection tface_landmarks;
		tface_landmarks = pose_model(img_t, tfaces[0]);
		auto tpoints = vectorize_landmarks(tface_landmarks);
		Rect rect_face_t = boundingRect(tpoints);
		// use only the face part 
		Mat face_t;
		target0(rect_face_t).copyTo(face_t);
		cv_image<bgr_pixel> img_s(source0);
		cv::Mat source;
		source0.convertTo(source, CV_32FC3);

		// detect source 
		std::vector<dlib::rectangle> sfaces = detector(img_s);
		if (sfaces.size()==0){
			continue;
		}
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

		//----------------------------------------------------------------------------------------------------------
		// Part II: warp target face to source face in order to get the expression of the source face 
		//          and get some morphing: eg. copy the mouth part to the target image 
		//----------------------------------------------------------------------------------------------------------
		Mat face_t2 = Mat::zeros(face_s.rows,face_s.cols, CV_32FC3);
		// we use delaunay triangles to operate affine transform 
		// we work for every triangle
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
		

		//motify mouth 
		int mouth_contours[] = {60,61,62,63,64,65,67};
		std::vector<Point2f> mouth_faces;
		std::vector<Point> mouth_int;
		for (int i=0;i<7;i++){
			mouth_faces.push_back(Point2f(spoints[mouth_contours[i]].x - rect_face_s.x,spoints[mouth_contours[i]].y - rect_face_s.y));
			mouth_int.push_back(Point(spoints[mouth_contours[i]].x - rect_face_s.x,spoints[mouth_contours[i]].y - rect_face_s.y));
		}
		Mat mouth_mask = Mat::zeros(face_t2.rows,face_t2.cols,CV_32FC3);
		fillConvexPoly(mouth_mask, mouth_int, Scalar(1.0, 1.0, 1.0), 16, 0);
		Rect mouth = boundingRect(mouth_faces);
		Mat mouth_src,mouth_dst,new_t2;
		multiply(face_s,mouth_mask,mouth_src);
		multiply(face_t2,Scalar(1.0,1.0,1.0)-mouth_mask,mouth_dst);
		new_t2 = mouth_src + mouth_dst;
		
		//----------------------------------------------------------------------------------------------------------
		// Part III: warp image back to its face
		//----------------------------------------------------------------------------------------------------------
		// warp image back to its face
		int face_contours[] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,26,25,24,23,20,19,18,17};
		int month_contours[] = {60,61,62,63,64,65,67};
		int elbows[] = {17,18,19,20,21,22,23,24,25,26};
		cv_image<bgr_pixel> img_t2(t2);

		// Detect target
		std::vector<dlib::rectangle> t2faces = detector(img_t2);
		full_object_detection t2face_landmarks;
		t2face_landmarks = pose_model(img_t2, t2faces[0]);
		auto t2points = vectorize_landmarks(t2face_landmarks);
		Rect rect_face_t2 = boundingRect(t2points);
		//show_landmarks(t2, t2points);

		//get the face contour points for source face and target face to get the 
		std::vector<Point2f> contours_facet;
		std::vector<Point2f> contours_faces;
		std::vector<Point2f> contours_facet2;
		std::vector<Point> mask_facet;
		Point center;
		for (int i=0;i<25;i++){
			contours_facet.push_back(tpoints[face_contours[i]]);
			center.x = center.x+tpoints[face_contours[i]].x;
			center.y = center.y+tpoints[face_contours[i]].y;
			contours_facet2.push_back(t2points[face_contours[i]]);
			mask_facet.push_back(Point(tpoints[face_contours[i]].x,tpoints[face_contours[i]].y));
			contours_faces.push_back(Point2f(spoints[face_contours[i]].x - rect_face_s.x,spoints[face_contours[i]].y - rect_face_s.y));
		}
		center.x = center.x/25;
		center.y = center.y/25;
		
		// triangulation
		// use the target face to get triangles 
		std::vector<std::vector<int>> triangles_contours = get_triangles_from_target(target0, contours_facet, "face_contour_");
		
		Mat face_tnext = target.clone();
		for (std::vector<std::vector<int>>::iterator it = triangles_contours.begin(); it!=triangles_contours.end();++it){
			std::vector<Point2f> sr, tr;
			int a,b,c;
			a = (*it)[0];
			b = (*it)[1];
			c = (*it)[2];
			sr.push_back(contours_faces[a]);
			sr.push_back(contours_faces[b]);
			sr.push_back(contours_faces[c]);
			tr.push_back(contours_facet[a]);
			tr.push_back(contours_facet[b]);
			tr.push_back(contours_facet[c]);
			
			Rect rs1 = boundingRect(sr);
			Rect rt1 = boundingRect(tr);
			
			transfer_to_source(sr,tr,new_t2,face_tnext);
		}
		Mat face_tnext2;
		linear_face_contours(contours_facet, face_tnext,target,face_tnext2);
		
		Mat t_next;
		face_tnext2.convertTo(t_next, CV_8UC3);
		
		imshow("Change Expression", t_next);
		//display it all on the screen
		//waitKey(30);
		char c=(char)waitKey(5);
		if(c==27){
			break;
		}
	}
	}catch(...){
		cout<<"Something went wrong! Please try again!"<<endl;
	}
}