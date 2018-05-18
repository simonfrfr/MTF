#include "mtf.h"
#include <QVector>
#include <stdio.h>
#include <iostream>
#include <eigen3/Eigen/QR>
#include "opencv2/imgproc.hpp"
#include <fftw3.h>
#define REAL 0
#define IMAG 1

using namespace cv;
using namespace std;

//http://www.imatest.com/docs/sharpness/


//void MTF::polyfit(const Mat& src_x, const Mat& src_y, Mat& dst, int order)
//{
//    CV_Assert((src_x.rows>0)&&(src_y.rows>0)&&(src_x.cols==1)&&(src_y.cols==1)
//            &&(dst.cols==1)&&(dst.rows==(order+1))&&(order>=1));
//    Mat X = Mat::zeros(src_x.rows, order+1,CV_64F);
//    Mat copy;
//    for(int i = 0; i <=order;i++)
//    {
//        copy = src_x.clone();
//        cv::pow(copy,i,copy);
//        copy.col(0).copyTo(X.col(i));
//    }
//    Mat W = ((X.t())*X).inv()*(X.t())*src_y;
//    W.copyTo(dst);
//}

void MTF::polyfit(const std::vector<double> &xv, const std::vector<double> &yv, std::vector<double> &coeff, int order)
{
    Eigen::MatrixXd A(xv.size(), order+1);
    Eigen::VectorXd yv_mapped = Eigen::VectorXd::Map(&yv.front(), yv.size());
    Eigen::VectorXd result;

    assert(xv.size() == yv.size());
    assert(xv.size() >= order+1);

    // create matrix
    for (size_t i = 0; i < xv.size(); i++)
    for (size_t j = 0; j < order+1; j++)
        A(i, j) = pow(xv.at(i), j);

    // solve for linear least squares fit
    result = A.householderQr().solve(yv_mapped);

    coeff.resize(order+1);
    for (size_t i = 0; i < order+1; i++)
        coeff[i] = result[i];
}
MTF::MTF()
{

}

Mat MTF::ExtractPatch(Mat sample, Point edge_start, Point edge_end, int desired_width, double crop_ratio){
    // Identify the edge direction.
    Mat patch;
    Point vec = edge_end - edge_start;
    // Discard both ends so that we MIGHT not include other edges
    // in the resulting patch.
    // TODO Auto-detect if the patch covers more than one edge!
    Point safe_start = (1 - crop_ratio) * edge_start + crop_ratio * edge_end;
    Point safe_end = crop_ratio * edge_start + (1 - crop_ratio) * edge_end;
    int minx = int(round(std::min(safe_start.x, safe_end.x)));
    int miny = int(round(std::min(safe_start.y, safe_end.y)));
    int maxx = int(round(std::max(safe_start.x, safe_end.x)));
    int maxy = int(round(std::max(safe_start.y, safe_end.y)));
    int ylb,yub,xub,xlb;
    if (abs(vec.x) > abs(vec.y)) { // near-horizontal edge
      ylb = max(0, miny - desired_width);
      yub = min(sample.rows, maxy + desired_width + 1);

      patch = sample(Rect(minx,ylb,maxx+1-minx,yub-ylb)).t();
    }
    else{  // near-vertical edge
      xlb = max(0, minx - desired_width);
      xub = min(sample.cols, maxx + desired_width + 1);
      patch = sample(Rect(xlb,miny,xub-xlb,maxy+1-miny)).t();

    }
    // Make sure white is on the left.

    if (patch.at<double>(0,0) < patch.at<double>(patch.rows-1,patch.cols-1)) {
      Mat patch2;
      flip(patch, patch2,1);
      patch = patch2;
    }

    // Make a floating point copy.
    patch.convertTo(patch, CV_64F);
    return patch;
}

tuple<double, Mat> MTF::FindEdgeSubPix(Mat patch, int desired_width){
  //Locates the edge position for each scanline with subpixel precision.
  int ph = patch.rows;
  int pw = patch.cols;
  // Get the gradient magnitude along the x direction.

  //Mat k_gauss = getGaussianKernel(15,1.0,CV_64F).t();
  //Mat temp;// = Mat::zeros(ph,pw,CV_64F);
  //filter2D(patch, temp,-1, k_gauss, Point(-1,-1),0,BORDER_REFLECT);
  //Mat hm = ahamming(10,5);
  //filter2D(temp, temp, -1, hm,Point(-1,-1),0, BORDER_REPLICATE);
  double jjj[1][2] = {{-1, 1.0}};
  Mat k_diff = Mat(1,2,CV_64F,jjj);
  Mat gradU;// = Mat::zeros(ph,pw,CV_64F);
  Mat grad;// = Mat::zeros(ph,pw,CV_64F);
  filter2D(patch, grad, -1, k_diff,Point(-1,-1),0, BORDER_REPLICATE);
  //grad = abs(gradU);
  //imshow("PostGradROI",grad);
  //waitKey();
  // Estimate subpixel edge position for each scanline.
  std::vector<double> ys = vector<double>(ph); // Need to make this now have sequantial numbers
  std::vector<double> xs = vector<double>(ph);
  std::vector<double> x_dummy = vector<double>(pw);  // Need to make this now have sequantial numbers

  for (int i = 0; i < ph; i++) {
      ys[i] = ((double) i);
  }
  for (int i = 0; i < pw; i++)
      x_dummy[i] = ((double) i);
  for (int y = 0; y <ph; y++) {
    // 1st iteration.
    double b=0;
    double a=0;
    for (int i = 1; i < pw; i++) {
        //if (y == 32)
        //    std::cout <<"" << y << "," << x_dummy[i] * grad.at<double>(y,i) <<""<< std::endl;
     double jjkl = grad.at<double>(y,i);
     b += x_dummy[i] * jjkl;
     a += grad.at<double>(y,i);
    }
    int c = (int)(round(b / a));
    //if (y == 32)
    //    std::cout <<":(c" << c <<")"<< std::endl;


    // 2nd iteration due to bias of different num of black and white pixels.
    int dw = std::min(std::min(c, desired_width), pw - c - 1);
    a = 0;
    b = 0;
    for (int i = c - dw; i < c + dw + 1; i++) {
        b += x_dummy[i] * grad.at<double>(y,i);
        a += grad.at<double>(y,i);
    }
            xs[y] = (int)(round(b / a));
  }

  //normalize(grad, grad, 0, 1, NORM_MINMAX);
  //imshow("GRAD",grad);
  //for (int i = xs.size()-8; i < xs.size(); i++)
  //    std::cout <<"(" << xs[i] << "," << ys[i] <<")"<< std::endl;
  double min, max;
  cv::minMaxLoc(xs, &min, &max);
  cout << min << ","<< max << endl;
  // Fit a second-order polyline for subpixel accuracy.
  vector<double> fitted_line = vector<double>(2);
  std::cout << xs.size() << "," << ys.size() <<std::endl;
  polyfit(ys, xs, fitted_line, 1);
  vector<double> fitted_parabola = vector<double>(3);
  polyfit(ys, xs, fitted_parabola, 2);
  double angle = atan(fitted_line[1]);
  std::cout <<"(Linear Fit:"<< fitted_line[1] <<")"<< std::endl;

  std::cout <<"(Parabolic Fit:"<< fitted_parabola[1] <<")"<< std::endl;


  //pb = np.poly1d(fitted_parabola)
  Mat centers = Mat(ph,1, CV_64F);
  for (int i = 0; i < ph; i++) {
    centers.at<double>(i,0) = (fitted_parabola[2]*((double)(i*i))) + (fitted_parabola[1]*((double)(i))) + fitted_parabola[0];
  }
  cv::minMaxLoc(centers, &min, &max);
  cout << min << ","<< max << endl;
  return make_tuple(angle, centers); //angle;//, centers
}
tuple<Mat, QVector<double>> MTF::AccumulateLine(Mat patch, Mat centers){
  //Adds up the scanlines along the edge direction.
  int ph = patch.rows;
  int pw = patch.cols;
  // Determine the final line length.
  double min, max;
  cv::minMaxLoc(centers, &min, &max);
  int w = std::min((int)(round(min)), pw - (int)(round(max)) - 1);
  int w4 = 2 * w + 1;
  // Accumulate a 4x-oversampled line.
  Mat psf4x = Mat(4,w4, CV_64F);
  psf4x = Mat::zeros(4, w4, CV_64F);
  Mat counts = Mat(4,1, CV_64F);
  counts = Mat::zeros(4,1, CV_64F);
  for (int y = 0; y < ph; y++){
    int ci = int(round(centers.at<double>(y,0)));
    int idx = 3 + 4 * ci - (int)(4 * centers.at<double>(y,0) + 2);
    for (int i = (ci - w); i < (ci + w + 1);i++) {
        int j = i-(ci - w);
        psf4x.at<double>(idx,j) += patch.at<double>(y,i);
    }
    counts.at<double>(idx,0) += 1;

  }
  //for (int i = 0; i < 4; i++)
  //    cout<<"C:"<<counts.at<double>(i,0) << endl;

  for (int i = 0; i < counts.rows;i++){
      psf4x.row(i)/=counts.at<double>(i,0);

  }

  //Mat hm = ahamming(psf4x.cols/18,(psf4x.cols+1)/36).t();
  //filter2D(psf4x, psf4x, -1, hm,Point(-1,-1),0, BORDER_REPLICATE);
  //psf4x /= counts;
  //for (int i = 0; i < psf4x.cols-1; i++)
  //    cout<<"D:"<<psf4x.at<double>(0,i) << endl;
  Mat psf2;
  psf2 = psf4x.t();
  Mat psfx;
  psfx = psf2.reshape(1,psf2.rows*psf2.cols);
  Mat psf = Mat(psfx.rows,1,CV_64F);
  psf = Mat::zeros(psfx.rows,1,CV_64F);
  QVector<double> Step = QVector<double>(psfx.rows-1);
  for (int i = 0; i < psfx.rows-1; i++) {
      psf.at<double>(i,0) = abs(psfx.at<double>(i+1,0) - psfx.at<double>(i,0));
      Step[i] = psfx.at<double>(i,1);
     // cout<<"E:"<<psfx.at<double>(i,0) << endl;
     //  cout<<"G:"<<psf.at<double>(i,0) << endl;
  }
  Mat hm2 = ahamming(psf.rows,(psf.rows+1)/2);
  psf = psf.mul(hm2);
  /*for (int i = 0; i < psfx.rows-1; i++) {
      cout << psf.at<double>(i,0) <<"," << endl;
  }*/
  return make_tuple(psf,Step);
}
Mat MTF::ahamming(int n, int mid) {
    // function generates a general asymmetric Hamming-type window
    // array. If mid = (n+1)/2 then the usual symmetric Hamming
    // window is returned
    //  n = length of array
    //  mid = midpoint (maximum) of window function
    //  data = window array (nx1)

    Mat data = Mat(n,1,CV_64F);
    data = Mat::zeros(n,1,CV_64F);

    double wid1 = mid-1;
    double wid2 = n-mid;
    double wid = max(wid1, wid2);
    double pie = M_PI;
    for (int i = 1; i < n; i++) {
        double arg = i-mid;
        data.at<double>(i,0) = cos( pie*arg/(wid) );
    }
    return 0.54 + 0.46*data;
}
double mag(fftw_complex j){
    return sqrt(j[REAL]*j[REAL]+j[IMAG]*j[IMAG]);
}

tuple<Mat,Mat> MTF::GetResponse(Mat psf, double angle){
  //Composes the MTF curve.


  // Compute FFT.

  int w = psf.rows;
  int N = w;
  // Set N to the number of complex elements in the input array
  fftw_complex *out;
  double *in;
  in = (double *)fftw_malloc(sizeof(double) * N);
  out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N);
  QVector<double> magnitude(w);
  // Initialize 'in' with N complex entries
  fftw_plan my_plan;
  //plan = fftw_plan_r2r_1d(N, in, out, FFTW_R2HC, FFTW_FORWARD);

  //fftw_plan_dft_r2c_1d(N, in, comout, FFTW_EXHAUSTIVE);
  my_plan = fftw_plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);
  for (int i = 0; i < N; i++){
      if (i < w)
        in[i] = psf.at<double>(i,0);
      else
        in[i] = 0;
      out[i][REAL] = 0;
      out[i][IMAG] = 0;
  }


  //cplan = fftw_plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);
  fftw_execute(my_plan);
  //fftw_execute(plan); /* repeat as needed */
  //fftw_execute(cplan); /* repeat as needed */
  // Use 'out' for something
  for (int i = 0; i < w; i++) {
      //cout << out[i][REAL] <<" + " << out[i][IMAG] <<"i" << endl;
      magnitude[i] = mag(out[i]);
  }
  fftw_destroy_plan(my_plan);
  fftw_free(in);
  fftw_free(out);
  // Slant correction factor.
  double slant_correction = cos(angle);
  // Compose MTF curve.
  // Normalize the low frequency response to 1 and compensate for the
  // finite difference.
  int rw = w / 4 + 1;
  double tempD = magnitude[3];
  //cout << tempD << endl;
  Mat freqs = Mat(rw, 1, CV_64F);
  Mat attns = Mat(rw, 1, CV_64F);
  cout <<"Correction: "<< 4.0 / ((double) N) / slant_correction << endl;
  for (int i =0; i < rw;i++) {
    magnitude[i] /= tempD;
    attns.at<double>(i,0) = ((double) i+3) * 4.0 / ((double) N) / slant_correction;
    double sinc = (sin(M_PI*((double) i+3)/((double) w))/(M_PI*((double) i)/((double) w)));
    freqs.at<double>(i,0) = (i+3 >= rw)?0:magnitude[i+3]; // /sinc/sinc;
    //cout << "<"<<magnitude.at<double>(i,0)<<","<<freqs.at<double>(i,0)<<"," <<attns.at<double>(i,0) <<">" << endl;
  }
  normalize(freqs, freqs, 0, 1, CV_MINMAX);

  return make_tuple(freqs, attns);
}
double MTF::FindMTF50P(Mat freqs,Mat attns,bool use_50p){
  //Locates the MTF50 given the MTF curve.
  //double min, max;
  //Point minPoint,maxPoint;
  //cv::minMaxLoc(attns, &min, &max,&minPoint,&maxPoint);
  double peak50 = 0;// ((use_50p)?max : 1.0) / 2.0;
  for (int i = 0; i < freqs.rows; i++){
      if (freqs.at<double>(i,0) > 0.5 &&attns.at<double>(i,0) < 0.5)
          peak50 = attns.at<double>(i,0);
  }  return peak50;//freqs.at<double>(maxPoint.y-1,0) + (freqs.at<double>(maxPoint.y,0) - freqs.at<double>(maxPoint.y-1,0)) * ratio;
}
tuple<bool,QVector<double>,QVector<double>,QVector<double>,QVector<double>,QVector<double>,QVector<double>,QVector<double>,QVector<double>> MTF::Compute(Mat sample, Point edge_start, Point edge_end, bool use_50p){
  /*Computes the MTF50P value of an edge.
  This function implements the slanted-edge MTF calculation method similar to
  the Imatest software. For more information, please visit
  http://www.imatest.com/docs/sharpness/.
    sample: The test target image.
    edge_start: The start point of edge.
    edge_end: The end point of edge.
    use_50p: Compute MTF50 value.
  Returns
    0 The MTF50/50P value.
    1, 2: The MTF curve
  */
    bool flipped = false;
  //Mat patch = ExtractPatch(sample, edge_start, edge_end, desired_width, crop_ratio);
  cout << "Passed Patch" << endl;
  Mat patch = Mat(sample.rows,sample.cols,CV_64F);
  sample.convertTo(patch, CV_64F);
  if (patch.at<double>(0,0) < patch.at<double>(patch.rows-1,patch.cols-1)) {
    Mat patch2;
    flip(patch, patch2,1);
    patch = patch2;
    flipped = true;
  }

  tuple<double,Mat> res1 = FindEdgeSubPix(patch, 60);
  double angle = get<0>(res1);
  Mat centers  = get<1>(res1);
  cout << "Passed FindEdge" << endl;
  tuple<Mat, QVector<double>> psff = AccumulateLine(patch, centers);
  Mat psf = get<0>(psff);
  cout << "Passed AccumulateLine" << endl;
  tuple<Mat,Mat> res2 = GetResponse(psf, angle);
  cout << "Passed GetResponse" << endl;
  Mat freqs = get<0>(res2);
  Mat attns = get<1>(res2);
  QVector<double> f2 = QVector<double>(freqs.rows);
  QVector<double> a2 = QVector<double>(freqs.rows);
  for (int i = 0; i < f2.size(); i++) {
      f2[i]=freqs.at<double>(i,0);
      a2[i]=attns.at<double>(i,0);
  }

  QVector<double> f3 = QVector<double>(psf.rows);
  QVector<double> a3 = QVector<double>(psf.rows);
  for (int i = 0; i < f3.size(); i++) {
      f3[i]=psf.at<double>(i,0);//freqs.at<double>(i,0);
      a3[i]=i;
  }
  QVector<double> f4 = QVector<double>(centers.rows);
  QVector<double> a4 = QVector<double>(centers.rows);
  for (int i = 0; i < f4.size(); i++) {
      f4[i]=centers.at<double>(i,0);//freqs.at<double>(i,0);
      a4[i]=i;
  }
  QVector<double> match = get<1>(psff);
  QVector<double> f5 = QVector<double>(match.length());
  QVector<double> a5 = QVector<double>(match.length());
  for (int i = 0; i < f5.size(); i++) {
      f5[i]=match[i];//freqs.at<double>(i,0);
      a5[i]=i;
  }
  double MTF = FindMTF50P(freqs, attns, use_50p);
  std::cout << "MTF50: " << MTF << endl;
  return make_tuple(flipped, f2, a2,f3,a3,f4,a4,f5,a5);
}
