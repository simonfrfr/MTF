#include "mainwindow.h"
#include <QApplication>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

int main(int argc, char *argv[]){
    const char* filename = "/home/simon/JuniorDesignOptical/test.jpg"; //canon_eos10d_sfr

    Mat I = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
    if( I.empty())
        return -1;

    QApplication a(argc, argv);
    MainWindow w(I);
    w.show();

    return a.exec();
}


