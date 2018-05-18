#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "mtf.h"
namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(cv::Mat img,QWidget *parent = 0);
    ~MainWindow();
    void sendData(QVector<double> x,QVector<double> y,QString XAxis,QString YAxis,int plot);
    void fitLineMat(cv::Point initial,QVector<double> x,QVector<double> y, cv::Mat mat);
    void getRectangle(cv::Point Initial, cv::Point Final);

private slots:
    void on_actionOpen_triggered();

private:
    Ui::MainWindow *ui;
    cv::Mat I;
};

#endif // MAINWINDOW_H
