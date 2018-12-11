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
	ifstream ifs(filename);
	float x, y, z;
	while (ifs >> x >> y >> z) {
		std::vector<int> tri;
		tri.push_back(x);
		tri.push_back(y);
		tri.push_back(z);
		triangles.push_back(tri);
	}

	return triangles;
}

int search_point(std::vector<Point2f> &points, Point2f & point){
	int re = -1;
	for (size_t i =0 ; i< points.size();i++){
		if ((int)point.x == (int)points[i].x && (int)point.y ==(int)points[i].y){
			re = i;
			break;
		}
	}
	return re;
}

std::vector<std::vector<int>> get_triangles_from_target(cv::Mat &image, std::vector<cv::Point2f> &points){
	// Rectangle to be used with Subdiv2D
	Rect rect(0, 0, image.size().width, image.size().height);
	// Create an instance of Subdiv2D
	Subdiv2D subdiv(rect);
	// Insert points into subdiv
	for( std::vector<Point2f>::iterator it = points.begin(); it != points.end(); it++)
	{
		subdiv.insert(*it);
	}
	std::vector<Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);
	std::vector<std::vector<int>> tri_partition;

	ofstream myfile;
  	myfile.open ("monalisa_triangle.txt");
  	
  	
	for( size_t i = 0; i < triangleList.size(); i++ ){
		Point2f p1(triangleList[i][0],triangleList[i][1]);
		Point2f p2(triangleList[i][2],triangleList[i][3]);
		Point2f p3(triangleList[i][4],triangleList[i][5]);
		int a,b,c;
		a = search_point(points,p1);
		b = search_point(points,p2);
		c = search_point(points,p3);
		if (a>-1 && b>-1 && c>-1){
			std::vector<int> v;
			v.push_back(a);
			v.push_back(b);
			v.push_back(c);
			tri_partition.push_back(v);
			myfile << a<<" "<<b<<" "<<c<<endl;
		}
	}
	myfile.close();
	return tri_partition;
}

void show_landmarks(cv::Mat image, std::vector<cv::Point2f> &points){
	for (cv::Point2f p : points){
		circle(image, cvPoint(p.x, p.y), 3, cv::Scalar(0, 0, 255), -1);
	}
	imshow("landmark",image);
	waitKey(0);
}

void show_delauney(cv::Mat image, std::vector<cv::Point2f> &points, std::vector<std::vector<int>> &triangles){
	cv::Scalar delaunay_color(255, 255, 255);
	cv::Size size = image.size();
	cv::Rect rect(0, 0, size.width, size.height);
	for (std::vector<std::vector<int>>::iterator it = triangles.begin(); it!=triangles.end();++it){
		int a,b,c;
		a = (*it)[0];
		b = (*it)[1];
		c = (*it)[2];
		Point2f p1 = points[a];
		Point2f p2 = points[b];
		Point2f p3 = points[c];
		if (rect.contains(p1) && rect.contains(p2) && rect.contains(p3)) {
		cv::line(image, p1, p2, delaunay_color, 1, CV_AA, 0);
		cv::line(image, p2, p3, delaunay_color, 1, CV_AA, 0);
		cv::line(image, p3, p1, delaunay_color, 1, CV_AA, 0);
		}
	}
	imshow("delauney",image);
	waitKey(0);
}


// scale transform 

// get target affine transform triangle 
void get_affine_triangle(std::vector<Point2f> & s1r,std::vector<Point2f> & s2r,std::vector<Point2f> & t1r,Mat & src, Mat & output, Mat & allMask){
	// Given a pair of triangles, find the affine transform.
  	Mat warp = getAffineTransform(s1r, s2r);
	cout<<warp.at<float>(0,0)<<" "<<warp.at<float>(0,1)<<" "<<warp.at<float>(0,2)<<endl;
	cout<<warp.at<float>(1,0)<<" "<<warp.at<float>(1,1)<<" "<<warp.at<float>(1,2)<<endl;
	//image_window win_target, win_new;
	cout<<s1r[0].x<<" "<<s1r[0].y<<endl;
	cout<<s2r[0].x<<" "<<s2r[0].y<<endl;
	cout<<s1r[1].x<<" "<<s1r[1].y<<endl;
	cout<<s2r[1].x<<" "<<s2r[1].y<<endl;
	cout<<s1r[2].x<<" "<<s1r[2].y<<endl;
	cout<<s2r[2].x<<" "<<s2r[2].y<<endl;

	cout<<t1r[0].x<<" "<<t1r[0].y<<endl;
	cout<<t1r[1].x<<" "<<t1r[1].y<<endl;
	cout<<t1r[2].x<<" "<<t1r[2].y<<endl;
	cout<<warp.cols<<" "<<warp.rows<<endl;

	//win_target.set_image(src);
	for (int i=0;i<3;i++){
		Point2f p = s1r[i];
		float x = warp.at<float>(0,0)* p.x + warp.at<float>(0,1) *p.y + warp.at<float>(0,2);
		float y = warp.at<float>(1,0)* p.x + warp.at<float>(1,1) *p.y + warp.at<float>(1,2);
		cout<<x<<" "<<y<<endl;
	}
	Mat warpImage = Mat::zeros(src.rows*2,src.cols*2,src.type());

	warpAffine(src, warpImage, warp, warpImage.size(), INTER_LINEAR,
             BORDER_REFLECT_101);
	//win_new.set_image(warpImage);
	src.convertTo(src, CV_8U);
	cv::imshow("src",src);
	waitKey(0);
	warpImage.convertTo(warpImage,CV_8U);
	cv::imshow("warp",warpImage);
	waitKey(0);
	/*
	cout<<warp.at<float>(0,0)<<" "<<warp.at<float>(0,1)<<" "<<warp.at<float>(0,2)<<endl;
	cout<<warp.at<float>(1,0)<<" "<<warp.at<float>(1,1)<<" "<<warp.at<float>(1,2)<<endl;
	std::vector<Point2f> t2r;
	std::vector<Point> t2rint;
	for (int i=0;i<3;i++){
		Point2f p = t1r[i];
		float x = warp.at<float>(0,0)* p.x + warp.at<float>(0,1) *p.y + warp.at<float>(0,2);
		float y = warp.at<float>(1,0)* p.x + warp.at<float>(1,1) *p.y + warp.at<float>(1,2);
		cout<<x<<" "<<y<<endl;
		t2r.push_back(Point2f(x,y));
		t2rint.push_back(Point(x,y));
	}
	Rect rt2 = boundingRect(t2r);
	cout<<rt2.height<<" "<<rt2.width<<endl;

	Mat mask = Mat::zeros(rt2.height, rt2.width, CV_32FC3);
    fillConvexPoly(mask, t2rint, Scalar(1.0, 1.0, 1.0), 16, 0);
	
	Mat warpImage = Mat::zeros(rt2.height,rt2.width,src.type());

	warpAffine(src, warpImage, warp, warpImage.size(), INTER_LINEAR,
             BORDER_REFLECT_101);
	*/
	cout<<"bounding rect"<<endl;
	Rect rt1 = boundingRect(t1r);
	Rect rt2(rt1.x,rt1.y,rt1.width, rt1.height);
	Mat mask = Mat::zeros(warpImage.rows, warpImage.cols, CV_8UC3);
	cout<<"multiply"<<endl;
	multiply(warpImage, mask, warpImage);
	cout<<"multiply"<<endl;
    //multiply(output(rt2), Scalar(1.0, 1.0, 1.0) - mask, output(rt2));
	cout<<"output"<<endl;
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

void test(){
	Point2f srctri[3];
	Point2f dsttri[3];
	Mat warp( 2, 3, CV_32FC1);
   	srctri[0] = Point2f(24.0,38.0);
	srctri[1] = Point2f(0.0,15.0);
	srctri[2] = Point2f(11.0,0.0);
	dsttri[0] = Point2f(23.0,35.0);
	dsttri[1] = Point2f(0.0,14.0);
	dsttri[2] = Point2f(11.0,0.0);
	warp = getAffineTransform(srctri,dsttri);
	cout<<warp.at<float>(0,0)<<" "<<warp.at<float>(0,1)<<" "<<warp.at<float>(0,2)<<endl;
	cout<<warp.at<float>(1,0)<<" "<<warp.at<float>(1,1)<<" "<<warp.at<float>(1,2)<<endl;
}

int main(){
	//test();
	// Load face detection and pose estimation models.
    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor pose_model;
    deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

	// read target neutral triangle information 
	//std::vector<std::vector<int>> triangles = read_triangles("delaunay.txt");

	//------------------------------------------------------
	//read images 
	//------------------------------------------------------
	// read target image t (neutral)
	string filename_t("monalisa.jpg");
	cv::Mat target0 = cv::imread(filename_t);
	imshow("target",target0);
	waitKey(0);
	cv_image<bgr_pixel> img_t(target0);
	//array2d<bgr_pixel> img_t;
	//load_image(img_t, filename_t);
	//cv::Mat target = dlib::toMat(img_t);
	// convert Mat to float data type
	cv::Mat target;
	target0.convertTo(target, CV_32F);
	Mat output = target.clone();
	Mat allmask = Mat::zeros(target.size(), CV_32FC3);
	
	cout<<"read target image"<<endl;

	// Detect target
    std::vector<dlib::rectangle> tfaces = detector(img_t);
	full_object_detection tface_landmarks;
	//std::vector<Point2f> tpoints;
	tface_landmarks = pose_model(img_t, tfaces[0]);
		// std::vector<cv::Point2f>
	auto tpoints = vectorize_landmarks(tface_landmarks);
	show_landmarks(target0, tpoints);
	std::vector<std::vector<int>> triangles = get_triangles_from_target(target,tpoints);
	show_delauney(target0,tpoints,triangles);

	// read source image s1 (neutral)
	// read source image s2 (non-neutral)
	string filename_s1("source-neutral.jpg");
	string filename_s2("source-happy.jpg");
	array2d<bgr_pixel> img_s1;
	array2d<bgr_pixel> img_s2;
	load_image(img_s1, filename_s1);
	load_image(img_s2, filename_s2);
	cv::Mat source1 = dlib::toMat(img_s1);
	cv::Mat source2 = dlib::toMat(img_s2);
	source1.convertTo(source1,CV_32F);
	source2.convertTo(source2,CV_32F);

	cout<<"read source images"<<endl;

	std::vector<dlib::rectangle> s1faces = detector(img_s1);
	full_object_detection s1face_landmarks;
	//std::vector<Point2f> s1points;
	s1face_landmarks = pose_model(img_s1, s1faces[0]);
		// std::vector<cv::Point2f>
	auto s1points = vectorize_landmarks(s1face_landmarks);
	

	std::vector<dlib::rectangle> s2faces = detector(img_s2);
	full_object_detection s2face_landmarks;
	//std::vector<Point2f> s2points;
	s2face_landmarks = pose_model(img_s2, s2faces[0]);
		// std::vector<cv::Point2f>
	auto s2points = vectorize_landmarks(s2face_landmarks);

	cout<<"begin transformation"<<endl;
	for (std::vector<std::vector<int>>::iterator it = triangles.begin(); it!=triangles.end();++it){
		std::vector<Point2f> s1r, s2r, t1r, t2r;
		int a,b,c;
		a = (*it)[0];
		b = (*it)[1];
		c = (*it)[2];
		cout<<a<<" "<<b<<" "<<c<<endl;
		s1r.push_back(s1points[a]);
		s1r.push_back(s1points[b]);
		s1r.push_back(s1points[c]);
		cout<<a<<" "<<b<<" "<<c<<endl;
		s2r.push_back(s2points[a]);
		s2r.push_back(s2points[b]);
		s2r.push_back(s2points[c]);
		cout<<a<<" "<<b<<" "<<c<<endl;
		t1r.push_back(tpoints[a]);
		t1r.push_back(tpoints[b]);
		t1r.push_back(tpoints[c]);
		cout<<a<<" "<<b<<" "<<c<<endl;
		Rect rs1 = boundingRect(s1r);
		Rect rs2 = boundingRect(s2r);
		Rect rt1 = boundingRect(t1r);
		cout<<a<<" "<<b<<" "<<c<<endl;
		std::vector<Point2f> triangle1,triangle2,triangle3,triangle4;
		for (int i = 0; i < 3; i++) {
			triangle1.push_back(Point2f(s1r[i].x - rs1.x, s1r[i].y - rs1.y));
			triangle2.push_back(Point2f(s2r[i].x - rs2.x, s2r[i].y - rs2.y));
			triangle3.push_back(Point2f(t1r[i].x - rt1.x, t1r[i].y = rt1.y));
		}
		Mat t1_rect;
		target(rt1).copyTo(t1_rect);
		cout<<"begin affine transformation"<<endl;
		imshow("test",t1_rect);
		get_affine_triangle(triangle1,triangle2,triangle3,t1_rect,output,allmask);
	}
	Mat target2;
	target.convertTo(target2, CV_8UC3);

	cout<<"show the result"<<endl;
	Point2f center(center_of_points(tpoints));
	Point centerInt((int)center.x, (int)center.y);
	allmask.convertTo(allmask, CV_8UC1, 256);
	output.convertTo(output, CV_8UC3);
	seamlessClone(output, target2, allmask, centerInt, output, NORMAL_CLONE);
	imshow("Morphed Face", output);
	// Display it all on the screen
	waitKey(30);





	
	// detect face and extract the target face region

	// align source image s1 & s2

	// warp target image t according to s2
	// warp target image s1 according to s2

	// use DWT to make a smooth transfer 

}