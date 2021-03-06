#pragma once
#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void myCanny(InputArray _src, OutputArray _dst,
	double low_thresh, double high_thresh,
	int aperture_size);