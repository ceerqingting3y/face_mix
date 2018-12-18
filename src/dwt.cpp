#include "dwt.hpp"
using namespace std;
using namespace cv;

void cal_dwt(Mat & new_src, Mat & info, Mat & dst_show){
    info = Mat::zeros(new_src.rows,new_src.cols,CV_32F);
    dst_show = Mat::zeros(new_src.rows,new_src.cols,CV_8U);
    cout<<"convert to gray"<<endl;
    cvtColor(new_src,new_src,CV_RGB2GRAY);
    cout<<"convert to 32f"<<endl;
    new_src.convertTo(new_src,CV_32F);
    cout<<"cvborder"<<endl;
    Mat im1,im2,im3,im4,im5,im6;
    im1=Mat::zeros(new_src.rows/2,new_src.cols,CV_32F);
    im2=Mat::zeros(new_src.rows/2,new_src.cols,CV_32F);
    im3=Mat::zeros(new_src.rows/2,new_src.cols/2,CV_32F);
    im4=Mat::zeros(new_src.rows/2,new_src.cols/2,CV_32F);
    im5=Mat::zeros(new_src.rows/2,new_src.cols/2,CV_32F);
    im6=Mat::zeros(new_src.rows/2,new_src.cols/2,CV_32F);
    cout<<"cvborder end"<<endl;
    float a,b,c,d;
    cout<<"check1"<<endl;
    cout<<new_src.rows<<" "<<new_src.cols<<endl;
    for(int i=0;i<new_src.rows;i+=2)
    {
        for(int j=0;j<new_src.cols;j++)
        {

            a=new_src.at<float>(i,j);
            b=new_src.at<float>(i+1,j);
            c=(a+b)*0.707;
            d=(a-b)*0.707;
            int _i=i/2;
            //cout<<i<<" "<<j<<endl;
            im1.at<float>(_i,j)=c;
            im2.at<float>(_i,j)=d;
        }
    }
    cout<<"check2"<<endl;

    for(int i=0;i<new_src.rows/2;i++)
    {
        for(int j=0;j<new_src.cols;j+=2)
        {

            a=im1.at<float>(i,j);
            b=im1.at<float>(i,j+1);
            c=(a+b)*0.707;
            d=(a-b)*0.707;
            int _j=j/2;
            im3.at<float>(i,_j)=c;
            im4.at<float>(i,_j)=d;
        }
    }

    cout<<"check3"<<endl;
    for(int i=0;i<new_src.rows/2;i++)
    {
        for(int j=0;j<new_src.cols;j+=2)
        {

            a=im2.at<float>(i,j);
            b=im2.at<float>(i,j+1);
            c=(a+b)*0.707;
            d=(a-b)*0.707;
            int _j=j/2;
            im5.at<float>(i,_j)=c;
            im6.at<float>(i,_j)=d;
        }
    }

    cout<<"check5"<<endl;
   
    cout<<"check6"<<endl;
    double min_im3, max_im3;
    cv::minMaxLoc(im3, &min_im3, &max_im3);
    Mat im3_scaled; //= Mat::zeros(im3.rows,im3.cols,CV_32F);
    cout<<"check61"<<endl;
    im3 = (im3 - min_im3)*255.0/(max_im3-min_im3);
    im3.convertTo(im3_scaled,CV_8U);
    cout<<"check62"<<endl;
    im3_scaled.copyTo(dst_show(Rect(0,0,im3_scaled.cols,im3_scaled.rows)));
    cout<<"check63"<<endl;
    im3.copyTo(info(Rect(0,0,im3.cols,im3.rows)));
    cout<<"check7"<<endl;
    double min_im4, max_im4;
    cv::minMaxLoc(im4, &min_im4, &max_im4);
    Mat im4_scaled=Mat::zeros(im4.rows,im4.cols,CV_32F);
    im4.convertTo(im4_scaled,CV_8U,255.0/(max_im4-min_im4),-255.0*min_im4/(max_im4-min_im4));
    im4_scaled.copyTo(dst_show(Rect(im4_scaled.cols-1,0,im4_scaled.cols,im4_scaled.rows)));
    im4.copyTo(info(Rect(im4.cols-1,0,im4.cols,im4.rows)));
    cout<<"check8"<<endl;
    double min_im5, max_im5;
    cv::minMaxLoc(im5, &min_im5, &max_im5);
    Mat im5_scaled = Mat::zeros(im5.rows,im5.cols,CV_32F);
    im5.convertTo(im5_scaled,CV_8U,255.0/(max_im5-min_im5),-255.0*min_im5/(max_im5-min_im5));
    im5_scaled.copyTo(dst_show(Rect(0,im5_scaled.rows-1,im5_scaled.cols,im5_scaled.rows)));
    im5.copyTo(info(Rect(0,im5.rows-1,im5.cols,im5.rows)));
    cout<<"check9"<<endl;
    double min_im6, max_im6;
    cv::minMaxLoc(im6, &min_im6, &max_im6);
    Mat im6_scaled =Mat::zeros(im6.rows,im6.cols,CV_32F);
    im6 = (im6 - min_im6)*255.0/(max_im6-min_im6);
    im6.convertTo(im6_scaled,CV_8U);
    im6_scaled.copyTo(dst_show(Rect(im6_scaled.cols-1,im6_scaled.rows-1,im6_scaled.cols,im6_scaled.rows)));
    im6.copyTo(info(Rect(im6_scaled.cols-1,im6_scaled.rows-1,im6.cols,im6.rows)));
}

void cal_idwt(Mat & src, Mat & info, Mat & dst_show){
    cout<<"check11"<<endl;
    Mat im3,im4,im5,im6,im11,im12,im13,im14;
    im11=Mat::zeros(src.rows/2,src.cols,CV_32F);
    im12=Mat::zeros(src.rows/2,src.cols,CV_32F);
    im13=Mat::zeros(src.rows/2,src.cols,CV_32F);
    im14=Mat::zeros(src.rows/2,src.cols,CV_32F);
    info = Mat::zeros(src.rows,src.cols,CV_32F);
    dst_show = Mat::zeros(src.rows,src.cols,CV_32F);
    float a,b,c,d;
    src(Rect(0,0,src.cols/2,src.rows/2)).copyTo(im3);
    src(Rect(src.cols/2-1,0,src.cols/2,src.rows/2)).copyTo(im4);
    src(Rect(0,src.rows/2-1,src.cols/2,src.rows/2)).copyTo(im5);
    src(Rect(src.cols/2-1,src.rows/2-1,src.cols/2,src.rows/2)).copyTo(im6);
    cout<<"check12"<<endl;
    for(int i=0;i<src.rows/2;i++)
    {
        for(int j=0;j<src.cols/2;j++)
        {
            int _j=j*2;
            im11.at<float>(i,_j)=im3.at<float>(i,j);     //Upsampling of stage I
            im12.at<float>(i,_j)=im4.at<float>(i,j);
            im13.at<float>(i,_j)=im5.at<float>(i,j);
            im14.at<float>(i,_j)=im6.at<float>(i,j);
        }
    }
    cout<<"check13"<<endl;
    for(int i=0;i<src.rows/2;i++)
    {
        for(int j=0;j<src.cols;j+=2)
        {

            a=im11.at<float>(i,j);
            b=im12.at<float>(i,j);
            c=(a+b)*0.707;
            im11.at<float>(i,j)=c;
            d=(a-b)*0.707;                           //Filtering at Stage I
            im11.at<float>(i,j+1)=d;
            a=im13.at<float>(i,j);
            b=im14.at<float>(i,j);
            c=(a+b)*0.707;
            im13.at<float>(i,j)=c;
            d=(a-b)*0.707;
            im13.at<float>(i,j+1)=d;
        }
    }
    cout<<"check14"<<endl;
    Mat temp=Mat::zeros(src.rows,src.cols,CV_32F);
    for(int i=0;i<src.rows/2;i++)
    {
        for(int j=0;j<src.cols;j++)
        {

            int _i=i*2;
            info.at<float>(_i,j)=im11.at<float>(i,j);     //Upsampling at stage II
            temp.at<float>(_i,j)=im13.at<float>(i,j); 
        }
    }
    cout<<"check14"<<endl;
    for(int i=0;i<src.rows;i+=2)
    {
        for(int j=0;j<src.cols;j++)
        {

            a=info.at<float>(i,j);
            b=temp.at<float>(i,j);
            c=(a+b)*0.707;
            info.at<float>(i,j)=c;                                      //Filtering at Stage II
            d=(a-b)*0.707;
            info.at<float>(i+1,j)=d;
        }
    }
    double min_dst,max_dst;
    //cv::minMaxLoc(info, &min_dst, &max_dst);
    //info.convertTo(dst_show,CV_8U,255.0/(max_dst-min_dst),-255.0*min_dst/(max_dst-min_dst));
    info.convertTo(dst_show,CV_8U);
}

void normalization(Mat & src, Mat & norm_src){
    int h = src.rows;
    int w = src.cols;
    norm_src = Mat::zeros(src.rows,src.cols,CV_32F);
    Rect r1 = Rect(0,0,w/2,h/2);
    Rect r2 = Rect(w/2-1,0,w/2,h/2);
    Rect r3 = Rect(0,h/2-1,w/2,h/2);
    Rect r4 = Rect(w/2-1,h/2-1,w/2,h/2);
    Mat m1,m2,m3,m4;
    src(r1).copyTo(m1);
    src(r2).copyTo(m2);
    src(r3).copyTo(m3);
    src(r4).copyTo(m4);
    cout<<"copy norm output"<<endl;
    float m1s = cv::sum(m1)[0];
    m1 = m1/m1s;
    float m2s = cv::sum(m2)[0];
    m2 = m2/m2s;
    float m3s = cv::sum(m3)[0];
    m3 = m3/m3s;
    float m4s = cv::sum(m4)[0];
    m4 = m4/m4s;
    cout<<"copy norm end"<<endl;
    m1.copyTo(norm_src(r1));
    m2.copyTo(norm_src(r2));
    m3.copyTo(norm_src(r3));
    m4.copyTo(norm_src(r4));
    cout<<"copy norm end!!!!"<<endl;


}
