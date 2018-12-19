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
	string filename_t = argv[3];
	cv::Mat target0 = cv::imread(filename_t);
	//imshow("target",target0);
	//waitKey(0);
	cv_image<bgr_pixel> img_t(target0);
	// convert Mat to float data type
	cv::Mat target;
	target0.convertTo(target, CV_32F);

	// Detect target
    std::vector<dlib::rectangle> tfaces = detector(img_t);
	full_object_detection tface_landmarks;
	tface_landmarks = pose_model(img_t, tfaces[0]);
	auto tpoints = vectorize_landmarks(tface_landmarks);

	
	Rect rect_face_t = boundingRect(tpoints);
	Mat face_t1;
	target0(rect_face_t).copyTo(face_t1);
	//imshow("face_t",face_t1);
	//waitKey(0);

	show_landmarks(target0, tpoints);
	//std::vector<std::vector<int>> triangles = read_triangles("jpg/baseline.txt");
	//get_triangles_from_target(target,tpoints,filename_t.substr(0,filename_t.find(".")));
	show_delauney(target0,tpoints,triangles);
	
	// read source image s1 (neutral)
	// read source image s2 (non-neutral)
	string filename_s1 = argv[1];
	string filename_s2 = argv[2];
	array2d<bgr_pixel> img_s1;
	array2d<bgr_pixel> img_s2;
	load_image(img_s1, filename_s1);
	load_image(img_s2, filename_s2);
	cv::Mat source10 = dlib::toMat(img_s1);
	cv::Mat source20 = dlib::toMat(img_s2);
	Mat source1,source2;
	source10.convertTo(source1,CV_32F);
	source20.convertTo(source2,CV_32F);
	

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

	Rect rect_face_s1 = boundingRect(s1points);
	Mat face_s1;
	source10(rect_face_s1).copyTo(face_s1);
	//imshow("face_s1",face_s1);
	//waitKey(0);

	Rect rect_face_s2 = boundingRect(s2points);
	Mat face_s2;
	source20(rect_face_s2).copyTo(face_s2);
	//imshow("face_s2",face_s2);
	//waitKey(0);

	//show_landmarks(source10, s1points);
	std::vector<std::vector<int>> triangles1 = get_triangles_from_target(source1,s1points,filename_s1.substr(0,filename_s1.find(".")));
	//show_delauney(source10,s1points,triangles1);

	//show_landmarks(source20, s2points);
	std::vector<std::vector<int>> triangles2 = get_triangles_from_target(source2,s2points,filename_s2.substr(0,filename_s2.find(".")));
	show_delauney(source20,s2points,triangles2);


	// try s1 to t1 transformation (show image )
	face_t1.convertTo(face_t1,CV_32FC3);
	face_s1.convertTo(face_s1,CV_32FC3);
	face_s2.convertTo(face_s2,CV_32FC3);
	//Mat face_t2 = face_t1.clone();
	// make the output face a little bigger 
	Mat face_t2 = Mat::zeros(face_t1.rows *1.5,face_t1.cols*1.5, CV_32FC3);
	Mat face_t3 = Mat::zeros(face_t1.rows *1.5,face_t1.cols*1.5, CV_32FC3);
	int l,w;
	if (face_s2.rows%2!=0){
		l = face_s2.rows+1;
	}
	if (face_s2.cols%2!=0){
		w = face_s2.cols+1;
	}
	Mat face_t4 = Mat::zeros(l,w, CV_32FC3);
	Mat face_t5 = Mat::zeros(l,w, CV_32FC3);
	//face_t2.convertTo(face_t2,CV_32FC3);
	Mat face_tmask = Mat::zeros(face_t1.size(), CV_32FC3);

	std::vector<std::vector<Point2f>> t2_points(68);
	for (int i =0;i<68;i++){
		std::vector<Point2f> new_ps;
		t2_points.push_back(new_ps);
	}
	cout<<"begin transformation"<<endl;
	for (std::vector<std::vector<int>>::iterator it = triangles.begin(); it!=triangles.end();++it){
		std::vector<Point2f> s1r, s2r, t1r, t2r;
		int a,b,c;
		a = (*it)[0];
		b = (*it)[1];
		c = (*it)[2];
		//cout<<a<<" "<<b<<" "<<c<<endl;
		s1r.push_back(s1points[a]);
		s1r.push_back(s1points[b]);
		s1r.push_back(s1points[c]);
		//cout<<a<<" "<<b<<" "<<c<<endl;
		s2r.push_back(s2points[a]);
		s2r.push_back(s2points[b]);
		s2r.push_back(s2points[c]);
		//cout<<a<<" "<<b<<" "<<c<<endl;
		t1r.push_back(tpoints[a]);
		t1r.push_back(tpoints[b]);
		t1r.push_back(tpoints[c]);
		//cout<<a<<" "<<b<<" "<<c<<endl;
		Rect rs1 = boundingRect(s1r);
		Rect rs2 = boundingRect(s2r);
		Rect rt1 = boundingRect(t1r);
		//cout<<a<<" "<<b<<" "<<c<<endl;
		std::vector<Point2f> triangle1,triangle2,triangle3;
		//turn the points to the coordinates in the cropped face image 
		std::vector<Point> t_int;
		for (int i = 0; i < 3; i++) {
			triangle1.push_back(Point2f(s1r[i].x - rect_face_s1.x, s1r[i].y - rect_face_s1.y));
			triangle2.push_back(Point2f(s2r[i].x - rect_face_s2.x, s2r[i].y - rect_face_s2.y));
			triangle3.push_back(Point2f(t1r[i].x - rect_face_t.x, t1r[i].y - rect_face_t.y));
			t_int.push_back(Point(t1r[i].x - rect_face_t.x, t1r[i].y - rect_face_t.y));
			//cout<<"t1r rect"<<rt1.x<<" "<<rt1.y<<endl;
			//cout<<t1r[i].x<<" "<<t1r[i].y<<endl;
		}

		transfer_to_source(triangle2,triangle3,face_s2,face_t3);
		transfer_to_source(triangle3,triangle2,face_t1,face_t4);
		transfer_to_source(triangle1,triangle2,face_s1,face_t5);
		//target_to_target(triangle3,triangle1, triangle2, t_int,face_t1,face_t5,face_s1,face_s2);

		Rect s11 = boundingRect(triangle1);
		Rect s22 = boundingRect(triangle2);
		Rect t11 = boundingRect(triangle3);
		std::vector<Point2f> s1ps,s2ps,t1ps;
		std::vector<Point2f>t2ps(3);
		// shift the points 
		for (int i = 0; i < 3; i++) {
			s1ps.push_back(Point2f(triangle1[i].x - s11.x, triangle1[i].y - s11.y));
			s2ps.push_back(Point2f(triangle2[i].x - s22.x, triangle2[i].y - s22.y));
			t1ps.push_back(Point2f(triangle3[i].x - t11.x, triangle3[i].y - t11.y));
			//t_int.push_back(Point(t1r[i].x - rect_face_t.x, t1r[i].y - rect_face_t.y));
			//cout<<"t1r rect"<<rt1.x<<" "<<rt1.y<<endl;
			//cout<<t1r[i].x<<" "<<t1r[i].y<<endl;
		}
		Mat warp = getAffineTransform(s1ps,s2ps);
		cv::transform(t1ps,t2ps,warp);
		
		// save t2 points in the initial figure coordinates 
		cout<<"rect face t "<<rect_face_t.x<<" "<<rect_face_t.y<<endl;
		cout<<"t11 "<<t11.x<<" "<<t11.y<<endl;
		t2_points[a].push_back(Point2f(t2ps[0].x+t11.x+rect_face_t.x, t2ps[0].y+t11.y+rect_face_t.y));
		t2_points[b].push_back(Point2f(t2ps[1].x+t11.x+rect_face_t.x, t2ps[1].y+t11.y+rect_face_t.y));
		t2_points[c].push_back(Point2f(t2ps[2].x+t11.x+rect_face_t.x, t2ps[2].y+t11.y+rect_face_t.y));
	}

	std::vector<Point2f> t2_new_points;
	calculate_new_points(t2_points,t2_new_points);

	for (std::vector<std::vector<int>>::iterator it = triangles.begin(); it!=triangles.end();++it){
		std::vector<Point2f> tt1,tt2;
		int a,b,c;
		a = (*it)[0];
		b = (*it)[1];
		c = (*it)[2];
		tt1.push_back(tpoints[a]);
		tt1.push_back(tpoints[b]);
		tt1.push_back(tpoints[c]);

		tt2.push_back(t2_new_points[a]);
		tt2.push_back(t2_new_points[b]);
		tt2.push_back(t2_new_points[c]);

		Rect rt1 = boundingRect(tt1);
		Rect rt2 = boundingRect(tt2);
		//cout<<a<<" "<<b<<" "<<c<<endl;
		std::vector<Point2f> tr1,tr2;
		//turn the points to the coordinates in the cropped face image 
		//std::vector<Point> tr_int;
		for (int i = 0; i < 3; i++) {
			tr1.push_back(Point2f(std::max(tt1[i].x - rect_face_t.x,float(0.0)), std::max(tt1[i].y - rect_face_t.y,float(0))));
			tr2.push_back(Point2f(std::max(tt2[i].x - rect_face_t.x,float(0.0)), std::max(tt2[i].y - rect_face_t.y,float(0))));
			//tr_int.push_back(Point(tt1[i].x - rect_face_t.x, tt1[i].y - rect_face_t.y));
			cout<<"points shifted "<<endl;
			cout<<tr2[i].x<<" "<<tr2[i].y<<endl;
		}

		cout<<"begin copy rect"<<endl;
		cout<<"begin affine transformation"<<endl;

		transfer_to_source(tr1,tr2,face_t1,face_t2);

	}
	cout<<"calculate new points! success!"<<endl;
	Mat t2;
	face_t2.convertTo(t2, CV_8UC3);
	//imshow("Morphed Face2", t2);
	//Display it all on the screen
	//waitKey(0);

	Mat t3;
	face_t3.convertTo(t3, CV_8UC3);
	//imshow("Morphed Face3", t3);
	//Display it all on the screen
	//waitKey(0);

	Mat t4;
	face_t4.convertTo(t4, CV_8UC3);
	//imshow("Morphed Face4", t4);
	//Display it all on the screen
	//waitKey(0);
	
	Mat info1,dst_show1;
	info1= Mat::zeros(t4.rows,t4.cols,CV_32F);
    dst_show1= Mat::zeros(t4.rows,t4.cols,CV_32F);
	cal_dwt(t4,info1,dst_show1);
	dst_show1.convertTo(dst_show1,CV_8U);
	imshow("DWT1",dst_show1);
	waitKey(0);

	Mat t5;
	face_t5.convertTo(t5, CV_8UC3);
	//imshow("Morphed Face5", t5);
	//Display it all on the screen
	//waitKey(0);

	Mat info2,dst_show2;
	info2= Mat::zeros(t5.rows,t5.cols,CV_32F);
    dst_show2= Mat::zeros(t5.rows,t5.cols,CV_32F);
	cal_dwt(t5,info2,dst_show2);
	dst_show2.convertTo(dst_show2,CV_8U);
	imshow("DWT2",dst_show2);
	waitKey(0);

	
	Mat info3,dst_show3;
	Mat face_s2_new = Mat::zeros(face_s2.rows,face_s2.cols,CV_32FC3);
	std::vector<Point> s2_new_points;
	for(int k =0;k<s2points.size();k++){
		s2_new_points.push_back(Point(s2points[k].x- rect_face_s2.x,s2points[k].y - rect_face_s2.y));
	}
	Rect face_s2_rect = boundingRect(s2_new_points);
	Mat masks2  = Mat::zeros(face_s2.rows,face_s2.cols,CV_32FC3);
	fillConvexPoly(masks2, s2_new_points, Scalar(1.0, 1.0, 1.0), 16, 0);
	
	multiply(face_s2,masks2,face_s2_new);
	copyMakeBorder(face_s2,face_s2_new,0,l-face_s2.rows,0,w-face_s2.cols,BORDER_CONSTANT);
	cout<<"face_s2_new: "<<face_s2_new.rows<<" "<<face_s2_new.cols<<endl;
	cal_dwt(face_s2_new,info3,dst_show3);
	dst_show3.convertTo(dst_show3,CV_8U);
	imshow("DWT3",dst_show3);
	waitKey(0);


	cout<<"info1"<<info1.rows<<" "<<info1.cols<<endl;
	cout<<"info2"<<info2.rows<<" "<<info2.cols<<endl;
	Mat subs = Mat::zeros(info1.rows,info1.cols,CV_32F);
	cout<<"subs"<<endl;
	cv::subtract(info3,info2,subs);
	
	// normalisation of subs
	Mat norm_subs;
	cout<<"normalization"<<endl;
	normalization(subs,norm_subs);

	subs.convertTo(subs,CV_8U);
	imshow("DWT4",subs);
	waitKey(0);

	Mat info4,dst_show4;
	float epsilon = 0.4;
	Mat merge;
	cout<<"calculate merge"<<endl;
	cout<<"info1 "<<info1.rows<<" "<<info1.cols<<endl;
	cout<<"norm_subs "<<norm_subs.rows<<" "<<norm_subs.cols<<endl;
	cout<<"info3 "<<info3.rows<<" "<<info3.cols<<endl;
	merge = (1-epsilon)*info1 - norm_subs.mul(info1)+ norm_subs.mul(info3) + epsilon* info3;
	cout<<"idwt"<<endl;
	cal_idwt(merge,info4,dst_show4);
	dst_show4.convertTo(dst_show4,CV_8U);
	imshow("DWT4",dst_show4);
	waitKey(0);
	
	// change histogram to make the face more like the original one
	Mat dst_final;
	Mat img1;
	cout<<"convert channel"<<endl;
	//t4.convertTo(t4,CV_8U);
	cout<<"convert color"<<endl;
	//cvtColor(t4,img1,CV_RGB2GRAY);
	t4.convertTo(img1,CV_8UC1);
	cout<<img1.dims<<endl;
	histogram_specify(dst_show4,img1,dst_final);
	cvtColor(dst_final,dst_final,CV_GRAY2RGB);
	imshow("Final result",dst_final);
	waitKey(0);
	/*
	Mat t3;
	face_t3.convertTo(t3, CV_8UC3);
	imshow("Morphed Face2", t3);
	// Display it all on the screen
	waitKey(0);
	
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
*/




	
	// detect face and extract the target face region

	// align source image s1 & s2

	// warp target image t according to s2
	// warp target image s1 according to s2

	// use DWT to make a smooth transfer 

}