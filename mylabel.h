#ifndef MYLABEL_H
#define MYLABEL_H
#include <QLabel>
#include <iostream>
#include "mainwindow.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

class mylabel: public QLabel
{
public:
    mylabel(QWidget *parent = 0);
    void setupMain(MainWindow * j);
    void mousePressEvent(QMouseEvent *event);
    void fitLineMat(Point initial,QVector<double> x,QVector<double> y);
private:
    int count = 0;
    MainWindow * main;
    cv::Point one;
    cv::Point two;

};

#endif // MYLABEL_H
