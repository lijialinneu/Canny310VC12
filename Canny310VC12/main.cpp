#include <opencv2/opencv.hpp>
#include <iostream>
#include <time.h>

using namespace cv;
using namespace std;

vector<Vec4f> operation(string path); // 对输入的图像进行直线检测

int main() 
{
	string path1 = "images/test3.jpg";
	string path2 = "images/test4.jpg";

	vector<Vec4f> lines_std1 = operation(path1);
	vector<Vec4f> lines_std2 = operation(path2);

	// 直线的聚类，延长，求出垂直、水平棱角线

	// 假设求出了轮廓，计算匹配度
	path1 = "images/test3_c.jpg";
	path2 = "images/test4_c.jpg";

	Mat image1 = imread(path1, IMREAD_GRAYSCALE);
	Mat image2 = imread(path2, IMREAD_GRAYSCALE);

	imshow("轮廓1", image1);
	imshow("轮廓2", image2);

	Moments m1, m2;

	vector<vector<Point>> contours1, contours2;
	findContours(image1, contours1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);//最外层轮廓  
	findContours(image2, contours2, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);//最外层轮廓

	//drawContours(copyImg1, contours1, -1, Scalar(0, 255, 0), 2, 8);
	//drawContours(copyImg1, contours1, -1, Scalar(0, 255, 0), 2, 8);

	//返回此轮廓与模版轮廓之间的相似度,a0越小越相似  
	double a0 = matchShapes(contours1[0], contours2[0], CV_CONTOURS_MATCH_I1, 0);
	cout << a0 << endl;

	waitKey(0);
	return 0;
}



/**
 * 图像滤波、边缘检测、直线检测操作
 * 返回检测后的直线
 */
vector<Vec4f> operation(string path)
{
	Mat image = imread(path, IMREAD_GRAYSCALE);
	
	blur(image, image, Size(3, 3)); // 使用3x3内核来降噪
	Canny(image, image, 50, 200, 3); // Apply canny edge

	// Create and LSD detector with standard or no refinement.
	// LSD_REFINE_NONE，没有改良的方式；
	// LSD_REFINE_STD，标准改良方式，将带弧度的线（拱线）拆成多个可以逼近原线段的直线度；
	// LSD_REFINE_ADV，进一步改良方式，计算出错误警告数量，通过增加精度，减少尺寸进一步精确直线。
	Ptr<LineSegmentDetector> ls = createLineSegmentDetector(LSD_REFINE_ADV, 0.99);
	double start = double(getTickCount());
	vector<Vec4f> lines_std;

	// Detect the lines
	ls->detect(image, lines_std);


	// Show found lines
	Mat drawnLines(image);
	Mat only_lines(image.size(), image.type());
	ls->drawSegments(drawnLines, lines_std);
	ls->drawSegments(only_lines, lines_std);
	imshow(path, drawnLines);
	imshow(path, only_lines);

	return lines_std;
}
