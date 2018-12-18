#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

class Histogramme 
{
    private:
    int channels[1];
    float hranges[2];
    const float * ranges[1];
    int histsize[1];
    public:
    Histogramme(){
        histsize[0] = 256;
        hranges[0] = 0.0;
        hranges[0] = 0.0;
        ranges[0] = hranges;
        channels[0] = 0;
    };
    MatND getHistogram(const Mat &img);
    void histogram_specify(const Mat &img1, const Mat &img2, Mat & dst);
};