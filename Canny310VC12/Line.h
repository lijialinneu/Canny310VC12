#pragma once
#include<opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

class Line
{
public:
	Vec4f line;    // 直线本身
	Point start;   // 直线的起点
	Point end;     // 直线的终点
	Point mid;     // 直线的中点
	double length; // 直线的长度
	double k;      // 直线的斜率
	double theta;  // 直线与水平方向的夹角


	Line(Vec4f line);
	Line(void);
	~Line(void);

private:
	double getLength(Line line);
	double getLength(Point start, Point end);
	Point getMidPoint();
};