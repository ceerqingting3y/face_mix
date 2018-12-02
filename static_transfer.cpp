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
#include "src/utils.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdlib.h>

using namespace dlib;
using namespace std;
using namespace cv;

// read predefined triangles 
std::vector<std::vector<int>> read_triangles(string filename){
	std::vector<std::vector<int>> triangles;
	ifstream ifs(fileName);
	float x, y;
	while (ifs >> x >> y >> z) {
		std::vector<int> tri;
		tri.push_back(x);
		tri.push_back(y);
		tri.push_back(z);
		triangles.push_back(tri);
	}

	return triangles;
}

// scale transform 

// get target affine transform triangle 
void get_affine_triangle(std::vector<Point2f> & s1r,std::vector<Point2f> & s2r,std::vector<Point2f> & t1r,Mat & src, Mat & output, Mat & allmask){
	// Given a pair of triangles, find the affine transform.
  	Mat warp = getAffineTransform(s1r, s2r);
	std::vector<Point2f> t2r;
	std::vector<Point> t2rint;
	for (int i=0;i<3;i++){
		Point2f p = t1r[i];
		float x = warp.at<float>(0,0)* p.x + warp.at<float>(0,1) *p.y;
		float y = warp.at<float>(1,0)* p.x + warp.at<float>(1,1) *p.y;
		t2r.push_back(Point2f(x,y));
		t2rint.push(Point(x,y));
	}
	Rect rt2 = boundingRect(t2r);

	Mat mask = Mat::zeros(rt2.height, rt2.width, CV_32FC3);
    fillConvexPoly(mask, t2rint, Scalar(1.0, 1.0, 1.0), 16, 0);
	
	Mat warpImage = Mat::zeros(rt2.height,rt2.width,src.type());

	warpAffine(src, warpImage, warp, warpImage.size(), INTER_LINEAR,
             BORDER_REFLECT_101);

	multiply(warpImage, mask, warpImage);
    multiply(output(rt2), Scalar(1.0, 1.0, 1.0) - mask, output(rt2));
    output(rt2) = output(rt2) + warpImage;
    allMask(rt2) += allMask(rt2) + mask;
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



int main(){

	// Load face detection and pose estimation models.
    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor pose_model;
    deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

	// read target neutral triangle information 
	std::vector<std::vector<int>> triangles = read_triangles("delaunay.txt");

	//------------------------------------------------------
	//read images 
	//------------------------------------------------------
	// read target image t (neutral)
	string filename_t("monalisa.jpg");
	Mat target = imread(filename_t);
	// convert Mat to float data type
	target.convertTo(target, CV_32F);
	Mat output = target.clone();
	Mat allmask = Mat::zeros(target.size(), CV_32FC3);

	// Detect target
    std::vector<dlib::rectangle> tfaces = detector(target);
	full_object_detection tface_landmarks;
	if (tfaces.size()>0){
		tface_landmarks = pose_model(target, tfaces[0]);
		// std::vector<cv::Point2f>
		auto tpoints = vectorize_landmarks(tface_landmarks);
	}

	// read source image s1 (neutral)
	// read source image s2 (non-neutral)
	string filename_s1("neutral_source.jpg");
	string filename_s2("move_source.jpg");
	Mat source1 = imread(filename_s1);
	Mat source2 = imread(filename_s2);
	source1.convertTo(source1,CV_32F);
	source2.convertTo(source2,CV_32F);

	std::vector<dlib::rectangle> s1faces = detector(source1);
	full_object_detection s1face_landmarks;
	if (s1faces.size()>0){
		s1face_landmarks = pose_model(source1, s1faces[0]);
		// std::vector<cv::Point2f>
		auto s1points = vectorize_landmarks(s1face_landmarks);
	}

	std::vector<dlib::rectangle> s2faces = detector(source2);
	full_object_detection s2face_landmarks;
	if (s2faces.size()>0){
		s2face_landmarks = pose_model(source2, s2faces[0]);
		// std::vector<cv::Point2f>
		auto s2points = vectorize_landmarks(s2face_landmarks);
	}

	for (std::vector<vector<int>>::iterator it = triangles.begin(); it!=triangles.end();++it){
		std::vector<Point2f> s1r, s2r, t1r, t2r;
		s1r.push_back(s1points[it[0]]);
		s1r.push_back(s1points[it[1]]);
		s1r.push_back(s1points[it[2]]);

		s2r.push_back(s2points[it[0]]);
		s2r.push_back(s2points[it[1]]);
		s2r.push_back(s2points[it[2]]);

		t1r.push_back(tpoints[it[0]]);
		t1r.push_back(tpoints[it[1]]);
		t1r.push_back(tpoints[it[2]]);

		Rect rs1 = boundingRect(s1r);
		Rect rs2 = boundingRect(s2r);
		Rect rt1 = boundingRect(t1r);

		std::vector<Point2f> triangle1,triangle2,triangle3,triangle4;
		for (int i = 0; i < 3; i++) {
			triangle1.push_back(Point2f(s1r[i].x - rs1.x, s1r[i].y - rs1.y));
			triangle2.push_back(Point2f(s2r[i].x - rs2.x, s2r[i].y - rs2.y));
			triangle3.push_back(Point2f(t1r[i].x - rt1.x, t1r[i].y = rt1.y));
		}
		Mat t1_rect;
		target(rt1).copyTo(t1_rect);
		get_affine_triangle(triangle1,triangle2,triangle3,t1_rect,output,allmask);
	}
	Mat target2;
	target.convertTo(target2, CV_8UC3);

	Point2f center(center_of_points(tpoints));
	Point centerInt((int)center.x, (int)center.y);
	mask.convertTo(mask, CV_8UC1, 256);
	output.convertTo(output, CV_8UC3);
	seamlessClone(output, target2, mask, centerInt, output, NORMAL_CLONE);
	imshow("Morphed Face", output);
	// Display it all on the screen
	waitKey(30);





	
	// detect face and extract the target face region

	// align source image s1 & s2

	// warp target image t according to s2
	// warp target image s1 according to s2

	// use DWT to make a smooth transfer 

}