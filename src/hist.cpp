#include "hist.hpp"
using namespace std;
using namespace cv;

MatND Histogramme::getHistogram(const Mat &img){
    MatND histo;
    calcHist(&img,1,channels,Mat(),histo,1,histsize,ranges);
    return histo;
}

void histogram_specify(const Mat &img1, const Mat &img2, Mat & dst){
    Histogramme hist0;
    cout<<"histo1"<<endl;
    MatND dst_hist = hist0.getHistogram(img2);
    
    cout<<"histo2"<<endl;
    MatND src_hist = hist0.getHistogram(img1);
    

    src_hist = src_hist/(img1.rows*img1.cols);
    dst_hist = dst_hist/(img2.rows*img2.cols);

    float src_cdf[256] = { 0 };
    float dst_cdf[256] = { 0 };
    src_cdf[0] = src_hist.at<float>(0);
    dst_cdf[0] = dst_hist.at<float>(0);
    for (int i = 1; i < 256; i++)
    {
        src_cdf[i] = src_cdf[i - 1] + src_hist.at<float>(i);
        dst_cdf[i] = dst_cdf[i - 1] + dst_hist.at<float>(i);
    }

    Mat m(1,256,CV_8U);
    float diff_cdf[256][256];
    for (int i = 0; i < 256; i++)
        for (int j = 0; j < 256; j++)
            diff_cdf[i][j] = fabs(src_cdf[i] - dst_cdf[j]);
    
    for (int i = 0; i < 256; i++)
    {
        float min = diff_cdf[i][0];
        int index = 0;
        for (int j = 1; j < 256; j++)
        {
            if (min > diff_cdf[i][j])
            {
                min = diff_cdf[i][j];
                index = j;
            }
        }
        m.at<uchar>(i) = static_cast<uchar>(index);
    }
    LUT(img1, m, dst);
}