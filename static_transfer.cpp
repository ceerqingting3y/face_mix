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

std::vector<std::vector<int>> get_triangles_from_target(cv::Mat &image, std::vector<cv::Point2f> &points, string filename){
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
  	myfile.open (filename+"triangle.txt");
  	
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
	//imshow("landmark",image);
	//waitKey(0);
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
	//imshow("delauney",image);
	//waitKey(0);
}


cv::Mat myGetAffineTransform(const std::vector<Point2f> & src, const std::vector<Point2f> &dst, int m)
{
    cv::Mat_<float> X = cv::Mat(m, 3, CV_32FC1, cv::Scalar(0));
    cv::Mat_<float> Y = cv::Mat(m, 2, CV_32FC1, cv::Scalar(0));

    for (int i = 0; i < m; i++)
    {
        float x0 = src[i].x, x1 = src[i].y;
        float y0 = dst[i].x, y1 = dst[i].y;

        X(i, 0) = x0;
        X(i, 1) = x1;
        X(i, 2) = 1;

        Y(i, 0) = y0;
        Y(i, 1) = y1;
    }

    cv::Mat_<float> F = (X.t()*X).inv()*(X.t()*Y);

    // cout << F << endl;

    return F.t();
}

void applyAffineTransform(Mat &warpImage, Mat &src, std::vector<Point2f> &srcTri, std::vector<Point2f> &dstTri)
{
    
    // Given a pair of triangles, find the affine transform.
	//scaling 
	//std::vector<Point2f> scale_src,scale_dst;
	//for (int i=0;i<3;i++){
	//	scale_src.push_back(Point2f(100*srcTri[i].x,100*srcTri[i].y));
	//	scale_dst.push_back(Point2f(100*dstTri[i].x,100*dstTri[i].y));
	//}
  	//Mat warp = getAffineTransform(scale_src,scale_dst);
    Mat warpMat = getAffineTransform( srcTri, dstTri );
	std::vector<Point2f> test_dst(3);
	cout<<"say something!"<<endl;
	cout<<src.size()<<" "<<test_dst.size()<<endl;
	cv::transform(srcTri,test_dst,warpMat);
	cout<<"not me!"<<endl;
	Mat mywarp = myGetAffineTransform(srcTri,dstTri,3);

	cout<<mywarp.at<float>(0,0)<<" "<<mywarp.at<float>(0,1)<<" "<<mywarp.at<float>(0,2)<<endl;
	cout<<mywarp.at<float>(1,0)<<" "<<mywarp.at<float>(1,1)<<" "<<mywarp.at<float>(1,2)<<endl;
	// Apply the Affine Transform just found to the src image
    warpAffine( src, warpImage, warpMat, warpImage.size(), INTER_LINEAR, BORDER_REFLECT_101);
	Mat row = cv::Mat::ones(1, 3, CV_64F);
	row.at<float>(0,0) = 0;
	row.at<float>(0,1) = 0;
	//warpMat.push_back(row);
	//Mat warp = warpMat.inv();
	Mat warp;
	invertAffineTransform(warpMat,warp);
	for (int i=0;i<3;i++){
		Point2f p = srcTri[i];
		float x = mywarp.at<float>(0,0)* p.x + mywarp.at<float>(0,1) *p.y + mywarp.at<float>(0,2);
		float y = mywarp.at<float>(1,0)* p.x + mywarp.at<float>(1,1) *p.y + mywarp.at<float>(1,2);
		cout<<"transformed dst point"<<int(x)<<" "<<int(y)<<endl;
		cout<<"correct dst point"<<int(srcTri[i].x)<<int(srcTri[i].y)<<endl;
		cout<<"test_dst"<<int(test_dst[i].x)<<" "<<int(test_dst[i].y)<<endl;

	}
    cout<<warp.at<float>(0,0)<<" "<<warp.at<float>(0,1)<<" "<<warp.at<float>(0,2)<<endl;
	cout<<warp.at<float>(1,0)<<" "<<warp.at<float>(1,1)<<" "<<warp.at<float>(1,2)<<endl;
	//cout<<warp.at<float>(2,0)<<" "<<warp.at<float>(2,1)<<" "<<warp.at<float>(2,2)<<endl;
	

    
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

void transfer_to_target(std::vector<Point2f> & t1_tri,std::vector<Point2f> & s1_tri, std::vector<Point2f> & s2_tri,std::vector<Point> & t_int, Mat & face_t1,Mat &face_t2, Mat & face_s1,Mat & face_s2){
	// Get mask by filling triangle
	cout<<"begin rect"<<endl;
	Rect r1 = boundingRect(t1_tri);
	Rect r = boundingRect(s1_tri);
	Rect rr = boundingRect(s2_tri);

	std::vector<Point2f> new_r1,new_r,new_rr;
	std::vector<Point> new_riint;
	for (int i = 0; i < 3; i++) {
			new_r1.push_back(Point2f(t1_tri[i].x - r1.x, t1_tri[i].y - r1.y));
			new_r.push_back(Point2f(s1_tri[i].x - r.x, s1_tri[i].y - r.y));
			new_rr.push_back(Point2f(s2_tri[i].x - rr.x, s2_tri[i].y - rr.y));
			//new_riint.push_back(Point(t1_tri[i].x - r1.x, t1_tri[i].y - r1.y));
			new_riint.push_back(Point(t1_tri[i].x - r1.x, t1_tri[i].y - r1.y));
			//cout<<"t1r rect"<<rt1.x<<" "<<rt1.y<<endl;
			//cout<<t1r[i].x<<" "<<t1r[i].y<<endl;
	}
	cout<<"begin mask"<<endl;
    Mat mask = Mat::zeros(r1.height, r1.width, CV_32FC3);
    fillConvexPoly(mask, new_riint, Scalar(1.0, 1.0, 1.0), 16, 0);
    
	cout<<"begin copy"<<endl;
    // Apply warpImage to small rectangular patches
    Mat img1Rect;
    //face_s1(r).copyTo(img1Rect);
	face_s2(rr).copyTo(img1Rect);
    
    Mat warpImage1 = Mat::zeros(r1.height, r1.width, img1Rect.type());
    //Mat warpImage2 = Mat::zeros(r.height, r.width, img2Rect.type());
    cout<<"begin affine"<<endl;
    applyAffineTransform(warpImage1, img1Rect, new_rr, new_r1);
    //applyAffineTransform(warpImage2, img2Rect, t2Rect, tRect);
    cout<<"end affine"<<endl;
    // Alpha blend rectangular patches
    //Mat imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2;
    
    // Copy triangular region of the rectangular patch to the output image
	cout<<"multiply1"<<endl;
    multiply(warpImage1,mask, warpImage1);
	cout<<"multiply2"<<endl;
    multiply(face_t2(r1), Scalar(1.0,1.0,1.0) - mask, face_t2(r1));
	cout<<"add"<<endl;
    face_t2(r1) = face_t2(r1) + warpImage1;
	//Mat temp;
	//face_t2.convertTo(temp,CV_8UC3);
	//imshow("morph",temp);
	//waitKey(0);
    
}

void transfer_to_source(std::vector<Point2f> & src_tri,std::vector<Point2f> & dst_tri,  Mat & src_face, Mat &dst_face){
	// Get mask by filling triangle
	// t1 --> s2 
	cout<<"begin rect"<<endl;
	Rect r1 = boundingRect(src_tri);
	Rect rr = boundingRect(dst_tri);

	std::vector<Point2f> new_r1,new_rr;
	std::vector<Point> new_riint;
	for (int i = 0; i < 3; i++) {
			new_r1.push_back(Point2f(src_tri[i].x - r1.x, src_tri[i].y - r1.y));
			new_rr.push_back(Point2f(dst_tri[i].x - rr.x, dst_tri[i].y - rr.y));
			new_riint.push_back(Point(dst_tri[i].x - rr.x, dst_tri[i].y - rr.y));
	}
	cout<<"begin mask"<<endl;
    Mat mask = Mat::zeros(rr.height, rr.width, CV_32FC3);
    fillConvexPoly(mask, new_riint, Scalar(1.0, 1.0, 1.0), 16, 0);
    
	cout<<"begin copy"<<endl;
    // Apply warpImage to small rectangular patches
    Mat img1Rect;
    //face_s1(r).copyTo(img1Rect);
	src_face(r1).copyTo(img1Rect);
    
    Mat warpImage1 = Mat::zeros(rr.height, rr.width, img1Rect.type());
    //Mat warpImage2 = Mat::zeros(r.height, r.width, img2Rect.type());
    cout<<"begin affine"<<endl;
    applyAffineTransform(warpImage1, img1Rect, new_r1, new_rr);
    //applyAffineTransform(warpImage2, img2Rect, t2Rect, tRect);
    cout<<"end affine"<<endl;
    // Alpha blend rectangular patches
    //Mat imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2;
    
    // Copy triangular region of the rectangular patch to the output image
	cout<<"multiply1"<<endl;
    multiply(warpImage1,mask, warpImage1);
	cout<<"multiply2"<<endl;
    multiply(dst_face(rr), Scalar(1.0,1.0,1.0) - mask, dst_face(rr));
	cout<<"add"<<endl;
    dst_face(rr) = dst_face(rr) + warpImage1;
	//Mat temp;
	//face_t2.convertTo(temp,CV_8UC3);
	//imshow("morph",temp);
	//waitKey(0);
    
}
void target_to_target(std::vector<Point2f> & t1_tri,std::vector<Point2f> & s1_tri, std::vector<Point2f> & s2_tri, std::vector<Point> & t_int, Mat & face_t1,Mat &face_t2, Mat & face_s1, Mat & face_s2){
	// input face_t2 is zeros 
	// Get mask by filling triangle
	cout<<"begin rect"<<endl;
	// boudning rect in the position of the initial image 
	Rect r1 = boundingRect(t1_tri);
	
	//Rect r2(r1.x,r1.y,min(face_t2.cols - r1.x, r1.width*2),min(face_t2.rows - r1.y,r1.height*2));
	Rect r2(r1.x,r1.y,r1.width,r1.height);
	cout<<r2.height + r2.y<<" "<<r2.width+r2.x<<endl;
	cout<<r1.height + r1.y<<" "<<r1.width+r1.x<<endl;
	Rect r = boundingRect(s1_tri);
	Rect rr = boundingRect(s2_tri);

	std::vector<Point2f> new_r1,new_r,new_rr; //coordinates shifted 
	std::vector<Point> new_riint;
	for (int i = 0; i < 3; i++) {
			new_r1.push_back(Point2f(t1_tri[i].x - r1.x, t1_tri[i].y - r1.y));
			new_r.push_back(Point2f(s1_tri[i].x - r.x, s1_tri[i].y - r.y));
			new_rr.push_back(Point2f(s2_tri[i].x - rr.x, s2_tri[i].y - rr.y));
			//new_riint.push_back(Point(t1_tri[i].x - r1.x, t1_tri[i].y - r1.y));
			new_riint.push_back(Point(t1_tri[i].x - r1.x, t1_tri[i].y - r1.y));
			//cout<<"t1r rect"<<rt1.x<<" "<<rt1.y<<endl;
			//cout<<t1r[i].x<<" "<<t1r[i].y<<endl;
	}
	// mask for t2 
	//cout<<"begin mask"<<endl;
    Mat mask = Mat::zeros(r1.height, r1.width, CV_32FC3);
    fillConvexPoly(mask, new_riint, Scalar(1.0, 1.0, 1.0), 16, 0);
    
	cout<<"begin copy"<<endl;
	// source image rect 
    Mat img1Rect;
    //face_s1(r).copyTo(img1Rect);
	face_t1(r1).copyTo(img1Rect);
	multiply(img1Rect,mask,img1Rect);

    Mat warpImage1 = Mat::zeros(r2.height, r2.width, img1Rect.type());
    //Mat warpImage2 = Mat::zeros(r.height, r.width, img2Rect.type());
    cout<<"begin affine"<<endl;
    applyAffineTransform(warpImage1, img1Rect, new_r, new_rr);
    //applyAffineTransform(warpImage2, img2Rect, t2Rect, tRect);
    cout<<"end affine"<<endl;
    // Alpha blend rectangular patches
    //Mat imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2;
    
    // Copy triangular region of the rectangular patch to the output image

	// get the image part 
	//cout<<"multiply1"<<endl;
    //multiply(warpImage1,mask, warpImage1);
	//cout<<"multiply2"<<endl;
    //multiply(face_t2(r1), Scalar(1.0,1.0,1.0) - mask, face_t2(r1));
	cout<<"add"<<endl;
	cout<<r2.height + r2.y<<" "<<r2.width+r2.x<<endl;
	cout<<r1.height + r1.y<<" "<<r1.width+r1.x<<endl;
	cout<<face_t2.rows<< " "<<face_t2.cols<<endl;
    face_t2(r2) = face_t2(r2) + warpImage1;
	//Mat temp;
	//face_t2.convertTo(temp,CV_8UC3);
	//imshow("morph",temp);
	//waitKey(0);
    
}

void static_transfer_target(std::vector<Point2f> &t1_tri, std::vector<Point2f> &t2_tri, Mat & face_t1, Mat & face_t2){
	// Get mask by filling triangle
	cout<<"begin rect"<<endl;
	cout<<t1_tri[0].x<<" "<<t1_tri[0].y<<endl;
	cout<<t1_tri[1].x<<" "<<t1_tri[1].y<<endl;
	cout<<t1_tri[2].x<<" "<<t1_tri[2].y<<endl;
	cout<<t2_tri[0].x<<" "<<t2_tri[0].y<<endl;
	cout<<t2_tri[1].x<<" "<<t2_tri[1].y<<endl;
	cout<<t2_tri[2].x<<" "<<t2_tri[2].y<<endl;
	Rect r1 = boundingRect(t1_tri);
	Rect r2= boundingRect(t2_tri);
	cout<<"r1 size "<<r1.x+r1.width<<" "<<r1.y+r1.height<<endl;
	cout<<"r2 size "<<r2.x+r2.width<<" "<<r2.y+r2.height<<endl;

	std::vector<Point2f> new_r1,new_r2;
	std::vector<Point> t2_int;
	for (int i = 0; i < 3; i++) {
			new_r1.push_back(Point2f(t1_tri[i].x - r1.x, t1_tri[i].y - r1.y));
			new_r2.push_back(Point2f(t2_tri[i].x - r2.x, t2_tri[i].y - r2.y));
			t2_int.push_back(Point(t2_tri[i].x - r2.x, t2_tri[i].y - r2.y));
			//cout<<"t1r rect"<<rt1.x<<" "<<rt1.y<<endl;
			//cout<<t1r[i].x<<" "<<t1r[i].y<<endl;
	}
	cout<<"begin mask"<<endl;
	// mask r1 src image patch 
    Mat mask = Mat::zeros(r2.height, r2.width, CV_32FC3);
    fillConvexPoly(mask, t2_int, Scalar(1.0, 1.0, 1.0), 16, 0);
    
	cout<<"begin copy"<<endl;
    // Apply warpImage to small rectangular patches
    Mat img1Rect;
	face_t1(r1).copyTo(img1Rect);
    
    Mat warpImage1 = Mat::zeros(r2.height, r2.width, img1Rect.type());
    //Mat warpImage2 = Mat::zeros(r.height, r.width, img2Rect.type());
    cout<<"begin affine"<<endl;
    applyAffineTransform(warpImage1, img1Rect, new_r1, new_r2);
    //applyAffineTransform(warpImage2, img2Rect, t2Rect, tRect);
    cout<<"end affine"<<endl;
    // Alpha blend rectangular patches
    //Mat imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2;
    
    // Copy triangular region of the rectangular patch to the output image
	cout<<"multiply1"<<endl;
    multiply(warpImage1,mask, warpImage1);
	
	cout<<"mask "<<mask.cols<<" "<<mask.rows<<endl;
	cout<<"r2 "<<r2.height<<" "<<r2.width<<endl;
	cout<<"r2 xy "<<r2.x<<" "<<r2.y<<endl;

	cout<<"multiply2"<<endl;
	try{
    	multiply(face_t2(r2), Scalar(1.0,1.0,1.0) - mask, face_t2(r2));
		cout<<"add"<<endl;
		face_t2(r2) = face_t2(r2) + warpImage1;
	}
	catch(...)
	{
		cout<<"WARNING!!!!!!!!!"<<endl;
	}
	
	//Mat temp;
	//face_t2.convertTo(temp,CV_8UC3);
	//imshow("morph",temp);
	//waitKey(0);
}

void calculate_new_points(std::vector<std::vector<Point2f>> & t2_points,std::vector<Point2f> &t2_new_points){
	for (int i=0;i<68;i++){
		if (t2_points[i].size()==0){
			cout<<"zero!!!!!!!"<<endl;
			t2_new_points.push_back(Point2f(0,0));
			continue;
		}
		Point2f p = std::accumulate(t2_points[i].begin(),t2_points[i].end(),Point2f(0,0));
		p.x = round(std::max(p.x/t2_points[i].size(),float(0.0)));
		p.y = round(std::max(p.y/t2_points[i].size(),float(0.0)));
		t2_new_points.push_back(p);
		cout<<t2_new_points[i].x<<" "<<t2_new_points[i].y<<endl;
	}	
}
int main(int argc, char** argv){
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
	string filename_t = argv[3];
	cv::Mat target0 = cv::imread(filename_t);
	//imshow("target",target0);
	//waitKey(0);
	cv_image<bgr_pixel> img_t(target0);
	//array2d<bgr_pixel> img_t;
	//load_image(img_t, filename_t);
	//cv::Mat target = dlib::toMat(img_t);
	// convert Mat to float data type
	cv::Mat target;
	target0.convertTo(target, CV_32F);
	//Mat output = target.clone();
	//Mat allmask = Mat::zeros(target.size(), CV_32FC3);
	
	cout<<"read target image"<<endl;

	// Detect target
    std::vector<dlib::rectangle> tfaces = detector(img_t);
	full_object_detection tface_landmarks;
	//std::vector<Point2f> tpoints;
	tface_landmarks = pose_model(img_t, tfaces[0]);
		// std::vector<cv::Point2f>
	auto tpoints = vectorize_landmarks(tface_landmarks);

	Rect rect_face_t = boundingRect(tpoints);
	Mat face_t1;
	target0(rect_face_t).copyTo(face_t1);
	//imshow("face_t",face_t1);
	//waitKey(0);

	show_landmarks(target0, tpoints);
	std::vector<std::vector<int>> triangles = read_triangles("jpg/baseline.txt");
	//get_triangles_from_target(target,tpoints,filename_t.substr(0,filename_t.find(".")));
	show_delauney(target0,tpoints,triangles);
	/*
	image_window win_faces;
	std::vector<full_object_detection> tttt;
	tttt.push_back(tface_landmarks);
	dlib::array<array2d<rgb_pixel>> face_chips;
	extract_image_chips(img_t, get_face_chip_details(tttt), face_chips);
	win_faces.set_image(tile_images(face_chips));
*/
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

		transfer_to_target(triangle3,triangle1,triangle2,t_int,face_t1,face_t3,face_s1,face_s2);
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

		static_transfer_target(tr1,tr2,face_t1,face_t2);

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