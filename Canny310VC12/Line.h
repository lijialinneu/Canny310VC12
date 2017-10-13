#pragma once
#include<opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

class Line
{
public:
	Vec4f line;    // ֱ�߱���
	Point start;   // ֱ�ߵ����
	Point end;     // ֱ�ߵ��յ�
	Point mid;     // ֱ�ߵ��е�
	double length; // ֱ�ߵĳ���
	double k;      // ֱ�ߵ�б��
	double theta;  // ֱ����ˮƽ����ļн�


	Line(Vec4f line);
	Line(void);
	~Line(void);

private:
	double getLength(Line line);
	double getLength(Point start, Point end);
	Point getMidPoint();
};