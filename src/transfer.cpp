#include "transfer.hpp"
using namespace std;
using namespace cv;
using namespace dlib;
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

// find the number of the point (between 0-67)
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

// use delauney method to get the triangles composition from image 
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

// give an images and its landmarks points, show it on the image
void show_landmarks(cv::Mat image, std::vector<cv::Point2f> &points){
	for (cv::Point2f p : points){
		circle(image, cvPoint(p.x, p.y), 3, cv::Scalar(0, 0, 255), -1);
	}
	//imshow("landmark",image);
	//waitKey(0);
}

// give an image, its landmark points, its triangles, show the triangles on the image 
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

// another version of getaffinetransform (use matrix / linear regression)
// the intrisic version of OPENCV is implemented by step method 
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
    return F.t();
}

// warpImage : the destination image 
// src: the source image 
// dsttri: the 3 points of a triangle that shows the position in the dst image 
void applyAffineTransform(Mat &warpImage, Mat &src, std::vector<Point2f> &srcTri, std::vector<Point2f> &dstTri)
{
    
    Mat warpMat = getAffineTransform( srcTri, dstTri );
    warpAffine( src, warpImage, warpMat, warpImage.size(), INTER_LINEAR, BORDER_REFLECT_101);

    /*
	std::vector<Point2f> test_dst(3);
	cout<<"say something!"<<endl;
	cout<<src.size()<<" "<<test_dst.size()<<endl;
	cv::transform(srcTri,test_dst,warpMat);
	cout<<"not me!"<<endl;
	Mat mywarp = myGetAffineTransform(srcTri,dstTri,3);
	cout<<mywarp.at<float>(0,0)<<" "<<mywarp.at<float>(0,1)<<" "<<mywarp.at<float>(0,2)<<endl;
	cout<<mywarp.at<float>(1,0)<<" "<<mywarp.at<float>(1,1)<<" "<<mywarp.at<float>(1,2)<<endl;
	// Apply the Affine Transform just found to the src image
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
	*/
}

// map src_face to dst_face according to the points affinetransform 
void transfer_to_source(std::vector<Point2f> & src_tri,std::vector<Point2f> & dst_tri,  Mat & src_face, Mat &dst_face){
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
    Mat img1Rect;
	src_face(r1).copyTo(img1Rect);
    
    Mat warpImage1 = Mat::zeros(rr.height, rr.width, img1Rect.type());
    cout<<"begin affine"<<endl;
    applyAffineTransform(warpImage1, img1Rect, new_r1, new_rr);
    cout<<"end affine"<<endl;
	cout<<"multiply1"<<endl;
    multiply(warpImage1,mask, warpImage1);
	cout<<"multiply2"<<endl;
    try{
        multiply(dst_face(rr), Scalar(1.0,1.0,1.0) - mask, dst_face(rr));
        cout<<"add"<<endl;
        dst_face(rr) = dst_face(rr) + warpImage1;
    }catch(...)
	{
		cout<<"WARNING!!!!!!!!! in transfer to source"<<endl;
	}
    
}

// carry out the target to target transform 
// using the mapping relation between source1 and source2 
void target_to_target(std::vector<Point2f> & t1_tri,std::vector<Point2f> & s1_tri, std::vector<Point2f> & s2_tri, std::vector<Point> & t_int, Mat & face_t1,Mat &face_t2, Mat & face_s1, Mat & face_s2){
	cout<<"begin rect"<<endl;
	Rect r1 = boundingRect(t1_tri);
    // the rect in face_t2 (same size as in face_t1)
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
			new_riint.push_back(Point(t1_tri[i].x - r1.x, t1_tri[i].y - r1.y));
	}
	// mask for t2 
	//cout<<"begin mask"<<endl;
    Mat mask = Mat::zeros(r1.height, r1.width, CV_32FC3);
    fillConvexPoly(mask, new_riint, Scalar(1.0, 1.0, 1.0), 16, 0);
    
	cout<<"begin copy"<<endl;
    Mat img1Rect;
	face_t1(r1).copyTo(img1Rect);
    // mask the outer area of the triangle 
	multiply(img1Rect,mask,img1Rect);

    Mat warpImage1 = Mat::zeros(r2.height, r2.width, img1Rect.type());
    cout<<"begin affine"<<endl;
    applyAffineTransform(warpImage1, img1Rect, new_r, new_rr);
    cout<<"end affine"<<endl;
	cout<<"add"<<endl;
	cout<<r2.height + r2.y<<" "<<r2.width+r2.x<<endl;
	cout<<r1.height + r1.y<<" "<<r1.width+r1.x<<endl;
	cout<<face_t2.rows<< " "<<face_t2.cols<<endl;
    face_t2(r2) = face_t2(r2) + warpImage1;
}


// t2_points contains each element a vector of point positions for one position 
// calculate the average of the points 
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