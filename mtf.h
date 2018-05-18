#ifndef MTF_H
#define MTF_H
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "qt5/QtCore/QVector"
#include <tuple>
#include <math.h>
using namespace cv;
using namespace std;

class MTF
{
public:
    MTF();
    void polyfit(const std::vector<double>& xv, const std::vector<double>& yv, std::vector<double>& coeff, int order);
    Mat ExtractPatch(Mat sample, Point edge_start, Point edge_end, int desired_width, double crop_ratio);
    tuple<double, Mat> FindEdgeSubPix(Mat patch, int desired_width);
    tuple<Mat, QVector<double>> AccumulateLine(Mat patch, Mat centers);
    tuple<Mat,Mat> GetResponse(Mat psf, double angle);
    double FindMTF50P(Mat freqs,Mat attns,bool use_50p);
    Mat ahamming(int n, int mid);
    tuple<bool,QVector<double>,QVector<double>,QVector<double>,QVector<double>,QVector<double>,QVector<double>,QVector<double>,QVector<double>> Compute(Mat sample, Point edge_start, Point edge_end, bool use_50p=true);
};

#endif // MTF_H
