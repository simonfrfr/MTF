#-------------------------------------------------
#
# Project created by QtCreator 2018-05-03T18:17:03
#
#-------------------------------------------------

QT       += core gui widgets printsupport

TARGET = JuniorDesignOptical
TEMPLATE = app

INCLUDEPATH += /usr/local/include/opencv
INCLUDEPATH += -I/usr/include/eigen3/
LIBS += -lfftw3
LIBS += -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
#DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0


SOURCES += \
        main.cpp \
        mainwindow.cpp \
        qcustomplot.cpp \
        mtf.cpp \
    mylabel.cpp

HEADERS += \
        mainwindow.h \
        qcustomplot.h \
        mtf.h \
    mylabel.h

FORMS += \
        mainwindow.ui
