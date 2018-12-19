#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

class Histogramme 
{
    private:
    int channels[1];
    int histsize[1];
    float hranges[2];
    const float * ranges[1];
    
    public:
    Histogramme(){
        channels[0] = 0;
        histsize[0] = 256;
        hranges[0] = 0.0;
        hranges[1] = 256.0;
        ranges[0] = hranges;
        
    };
    MatND getHistogram(const Mat &img);
};

void histogram_specify(const Mat &img1, const Mat &img2, Mat & dst);