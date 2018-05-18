#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
using namespace cv;
using namespace std;

MainWindow::MainWindow(Mat img, QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    // generate some data:
    QVector<double> x(101), y(101); // initialize with entries 0..100
    for (int i=0; i<101; ++i)
    {
      x[i] = i/50.0 - 1; // x goes from -1 to 1
      y[i] = x[i]*x[i]; // let's plot a quadratic function
    }
    // create graph and assign data to it:
    ui->plot->addGraph();
    ui->plot->graph(0)->setData(x, y);
    // give the axes some labels:
    ui->plot->xAxis->setLabel("x");
    ui->plot->yAxis->setLabel("y");
    // set axes ranges, so we see all data:
    ui->plot->xAxis->setRange(-1, 1);
    ui->plot->yAxis->setRange(0, 1);
    ui->plot->replot();
    I = img.clone();

    fitLineMat(Point(0,0),QVector<double>(),QVector<double>(), I);
    ui->Matlabel->setScaledContents(true);
    ui->Matlabel->setupMain(this);
    getRectangle(Point(),Point());
}

void MainWindow::sendData(QVector<double> x,QVector<double> y,QString XAxis,QString YAxis,int plot){
    double minx = x[1];
    double maxx = x[1];
    double miny = y[1];
    double maxy = y[1];
    QCustomPlot * plotP;
    plotP = ui->plot;
    if (plot == 2)
        plotP = ui->plot_2;
    else if (plot == 3)
        plotP = ui->plot_3;


    for (int i = 0; i < x.size();i++) {
        if (minx > x[i] && isfinite(x[i])) minx = x[i];
        if (miny > y[i] && isfinite(y[i])) miny = y[i];
        if (maxx < x[i] && isfinite(x[i])) maxx = x[i];
        if (maxy < y[i] && isfinite(y[i])) maxy = y[i];
    }
    // create graph and assign data to it:
    plotP->addGraph();
    plotP->graph(0)->setData(x, y);
    // give the axes some labels:
    plotP->xAxis->setLabel(XAxis);
    plotP->yAxis->setLabel(YAxis);
    // set axes ranges, so we see all data:
    plotP->xAxis->setRange(minx, maxx);
    plotP->yAxis->setRange(miny, maxy);
    plotP->replot();
}
void MainWindow::fitLineMat(Point initial,QVector<double> x,QVector<double> y, Mat mat){
    Mat mati;
    initial = Point(0,0);
    cvtColor(mat, mati, CV_GRAY2RGB);
    //for (int i = 0; i < x.length()-1; i++)
    //    line(mati, Point(x[i]+initial.y,y[i]+initial.x), Point(x[i+1]+initial.y,y[i+1]+initial.x), Scalar( 255, 0, 0 ));
    //line(mati, Point(initial.x,initial.y), Point(0,0), Scalar( 255, 0, 0 ));
    cout <<  mati.cols<<","<<mati.rows << endl;
    cout <<  initial.x<<","<<initial.y << endl;
    ui->Matlabel->setPixmap(QPixmap::fromImage(QImage(mati.data, mati.cols, mati.rows, mati.step, QImage::Format_RGB888)));
    ui->Matlabel->fitLineMat(initial,x,y);
}

void MainWindow::getRectangle(Point Initial, Point Final){
    MTF j;
    auto res = j.Compute(I, Initial, Final, true);
    if (get<0>(res)) {
        Mat patch2;
        flip(I, patch2,1);
        I = patch2;
    }
    sendData(get<2>(res),get<1>(res),QString("Frequency, Cycles/pixel"),QString("SFR (MTF)"),1);
    sendData(get<4>(res),get<3>(res),QString("4x Oversample Value"),QString("Occurances"),2);
    sendData(get<8>(res),get<7>(res),QString("Y (Pix)"),QString("X (Pix)"),3);
    fitLineMat(Initial,get<5>(res),get<6>(res), I);


}

MainWindow::~MainWindow()
{
    delete ui;
}




void MainWindow::on_actionOpen_triggered()
{
    QString fileName = QFileDialog::getOpenFileName(this,
            tr("Open Images"), "",
            tr("(*.png *.jpg *.bmp)"));
    if (fileName != "") {
        I = imread(fileName.toStdString().c_str(), CV_LOAD_IMAGE_GRAYSCALE);
        getRectangle(Point(),Point());
    }
}
