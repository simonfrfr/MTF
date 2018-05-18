#include "mylabel.h"
#include <QString>
#include <QMouseEvent>
#include <QPainter>
#include <QPixmap>
#include <QDebug>
#include <iostream>


mylabel::mylabel(QWidget *parent):
    QLabel(parent)
{

}
void mylabel::mousePressEvent(QMouseEvent *event)
{
    QString x = QString::number(event->x());
    QString y = QString::number(event->y());
    QPixmap pix = this->pixmap()->copy();
    QSize j = pix.size();
    QSize k = this->size();
    double ratioX = ((double)k.width())/((double)j.width());
    double ratioY = ((double)k.height())/((double)j.height());
      QPainter painter(&pix);
      QPen pen;
      pen.setColor("red");
      pen.setCapStyle(Qt::RoundCap);
      pen.setWidth(10);
      painter.setPen(pen);
      painter.drawPoint(event->x()/ratioX, event->y()/ratioY);
      painter.end(); // probably not needed
      this->setPixmap(pix);

      this->update();
      if (count ==0){
          one = cv::Point(event->x()/ratioX, event->y()/ratioY);
          count++;
      }
      else if (count == 1) {
        two = cv::Point(event->x()/ratioX, event->y()/ratioY);
        count = 0;
        main->getRectangle(Point(std::min(one.x,two.x),std::min(one.y,two.y)),Point(std::max(one.x,two.x),std::max(one.y,two.y)));
      }

   std::cout << x.toStdString() << "," << y.toStdString() << std::endl;
}
void mylabel::fitLineMat(Point initial,QVector<double> x,QVector<double> y){
    QPixmap pix = this->pixmap()->copy();
    QPainter painter(&pix);

    QPen pen;
    pen.setColor("red");
    pen.setCapStyle(Qt::RoundCap);
    pen.setWidth(4);
    painter.setPen(pen);
    for (int i = 0; i < x.length()-1; i++)
        painter.drawLine(x[i]+initial.y,y[i]+initial.x, x[i+1]+initial.y,y[i+1]+initial.x);
    painter.end(); // probably not needed
    this->setPixmap(pix);

    this->update();
}

void mylabel::setupMain(MainWindow * j){
    main = j;
}
