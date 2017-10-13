#include "Line.h"
#include<opencv2/opencv.hpp>
#include<math.h>

#define PI 3.1415926
#define MAX 100000

using namespace cv;

Line::Line(Vec4f line) {

	// ������㣬�յ���е�
	start = Point(line[0], line[1]);
	end = Point(line[2], line[3]);
	mid = getMidPoint();

	// ���㳤�Ⱥ���Գ���
	length = getLength(start, end);

	// ����ֱ�ߵ�б��
	if (end.x - start.x != 0) {
		k = (double)(start.y - end.y) / (start.x - end.x);
	}else {
		k = MAX;
	}

	//����ֱ����ˮƽ����ļн�
	theta = atan(k) * 180 / PI;
}

Line::Line(void) {
}

Line::~Line(void){
}

/**
 * ����ֱ�ߵĳ���
 */
double Line::getLength(Line line) {
	return getLength(line.start, line.end);
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

