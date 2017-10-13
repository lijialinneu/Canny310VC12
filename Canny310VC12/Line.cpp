#include "Line.h"
#include<opencv2/opencv.hpp>
#include<math.h>

#define PI 3.1415926


using namespace cv;

Line::Line(Vec4f line) {

	// ������㣬�յ���е�
	start = Point(line[0], line[1]);
	end = Point(line[2], line[3]);
	mid = getMidPoint();
	
	length = getLength(start, end); // ���㳤�Ⱥ���Գ���
	k = getK(); // ����ֱ�ߵ�б��
	theta = getTheta(); //����ֱ����ˮƽ����ļн�
}

Line::Line(void) {
}

Line::~Line(void){
}

/**
 * ����ֱ�ߵĳ���
 */
double Line::getLength() {
	return getLength(start, end);
}

double Line::getLength(Point start, Point end) {
	double x = start.x - end.x;
	double y = start.y - end.y;
	return sqrt(pow(x, 2) + pow(y, 2));
}

/**
 * ����ֱ�ߵ��е�
 */
Point Line::getMidPoint() {
	double mid_x = (start.x + end.x) / 2;
	double mid_y = (start.y + end.y) / 2;
	Point *mid = new Point(mid_x, mid_y);
	return *mid;
}

/**
 * ����б��
 */
double Line::getK() {
	double k = 100000;
	if (abs(end.x - start.x) > 1e-6) {
		k = (double)(start.y - end.y) / (start.x - end.x);
	}
	return k;
}


/**
 * ����theta
 */
double Line::getTheta() {
	return atan(k) * 180 / PI;
}