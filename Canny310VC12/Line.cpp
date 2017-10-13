#include "Line.h"
#include<opencv2/opencv.hpp>
#include<math.h>

#define PI 3.1415926
#define MAX 100000

using namespace cv;

Line::Line(Vec4f line) {

	// 计算起点，终点和中点
	start = Point(line[0], line[1]);
	end = Point(line[2], line[3]);
	mid = getMidPoint();

	// 计算长度和相对长度
	length = getLength(start, end);

	// 计算直线的斜率
	if (end.x - start.x != 0) {
		k = (double)(start.y - end.y) / (start.x - end.x);
	}else {
		k = MAX;
	}

	//计算直线与水平方向的夹角
	theta = atan(k) * 180 / PI;
}

Line::Line(void) {
}

Line::~Line(void){
}

/**
 * 计算直线的长度
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
 * 计算直线的中点
 */
Point Line::getMidPoint() {
	double mid_x = (start.x + end.x) / 2;
	double mid_y = (start.y + end.y) / 2;
	Point *mid = new Point(mid_x, mid_y);
	return *mid;
}

