#include <opencv2/opencv.hpp>
#include <iostream>
#include <time.h>
#include <set>
#include "Line.h"

using namespace cv;
using namespace std;

vector<Vec4f> operation(string path, Mat image); // 对输入的图像进行直线检测
vector<Line> createLine(vector<Vec4f> lines); // 构造直线

bool canCluster(Line l1, Line l2); // 判断能否聚合或连接
double distanceBetweenLine(Line l1, Line l2); // 估计两条直线间的距离
bool isPointNear(Point p1, Point p2, double th); // 判断两个点是否接近
int isConnect(Line l1, Line l2, int th); // 返回连接的类型

vector<Line> connectLines(vector<Line> lines, int th, Mat dst); // 连接直线
vector<Line> clusterLines(vector<Line> lines, int th, Mat dst); // 聚合直线
double match(vector<Vec4f> lines1, vector<Vec4f> lines2, 
	InputArray m1, InputArray m2); // 计算两组直线的匹配度


int main() {

	// Step1 提取直线
	string path1 = "images/test3.jpg";
	string path2 = "images/test4.jpg";

	Mat image1 = imread(path1, IMREAD_GRAYSCALE);
	Mat image2 = imread(path2, IMREAD_GRAYSCALE);
	
	vector<Vec4f> lines_std1 = operation(path1, image1);
	vector<Vec4f> lines_std2 = operation(path2, image2);



	// Step2 搞出轮廓，完全不知道咋整，艹
	// 直线的聚类，延长，求出垂直、水平棱角线

	double rate = match(lines_std1, lines_std2, image1, image2);
	cout << "匹配度是：" << rate << endl;

	vector<Vec4f>().swap(lines_std1);
	vector<Vec4f>().swap(lines_std2);

	// Step3 假设求出了轮廓，计算匹配度
	
	//path1 = "images/test3_c.jpg";
	//path2 = "images/test4_c.jpg";

	//Mat image1 = imread(path1, IMREAD_GRAYSCALE);
	//Mat image2 = imread(path2, IMREAD_GRAYSCALE);

	//image1 = 255 - image1; // 反色
	//image2 = 255 - image2;

	//imshow("1", image1);
	//imshow("2", image2);
	//
	//Mat image1_copy = imread(path1);
	//Mat image2_copy = imread(path2);

	//vector<vector<Point>> contours1, contours2;
	//findContours(image1, contours1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);//最外层轮廓  
	//findContours(image2, contours2, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);//最外层轮廓

	//drawContours(image1_copy, contours1, -1, Scalar(0, 255, 0), 2, 8);
	//drawContours(image2_copy, contours2, -1, Scalar(0, 255, 0), 2, 8);

	//imshow("轮廓1", image1_copy);
	//imshow("轮廓2", image2_copy);

	////返回此轮廓与模版轮廓之间的相似度,a0越小越相似  
	//double a0 = matchShapes(contours1[0], contours2[0], CV_CONTOURS_MATCH_I1, 0);
	//cout << a0 << endl;

	waitKey(0);
	return 0;
}



/**
 * 图像滤波、边缘检测、直线检测操作
 * 返回检测后的直线
 */
vector<Vec4f> operation(string path, Mat image) {
	blur(image, image, Size(3, 3)); // 使用3x3内核来降噪
	Canny(image, image, 50, 200, 3); // Apply canny edge

	// Create and LSD detector with standard or no refinement.
	// LSD_REFINE_NONE，没有改良的方式；
	// LSD_REFINE_STD，标准改良方式，将带弧度的线（拱线）拆成多个可以逼近原线段的直线度；
	// LSD_REFINE_ADV，进一步改良方式，计算出错误警告数量，通过增加精度，减少尺寸进一步精确直线。
	Ptr<LineSegmentDetector> ls = createLineSegmentDetector(LSD_REFINE_STD, 0.99);
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


/**
 * 对向量中的每一条直线构造Line对象
 * 返回一个向量集合，集合里的元素是Line对象
 */
vector<Line> createLine(vector<Vec4f> lines) {
	vector<Line> LineSet;
	size_t len = lines.size();
	for (int i = 0; i < len; i++) {
		Line *line = new Line(lines[i]);
		LineSet.push_back(*line);
	}
	vector<Vec4f>().swap(lines);
	return LineSet;
}


/**
 * 过滤短小直线
 * TODO 过滤孤立的直线
 */
vector<Line> cleanShort(vector<Line> lines) {
	vector<Line> *result = new vector<Line>();
	size_t length = lines.size();
	if (length == 0) return lines;

	// 计算average长度
	double sum = 0;
	for (int i = 0; i < length; i++) {
		Line line = lines[i];
		sum += line.length;
	}
	double avg = (sum / length) / 2;

	// 过滤短小的直线
	for (int i = 0; i < length; i++) {
		Line line = lines[i];
		if (line.length >= avg) {
			(*result).push_back(line);
		}		
	}
	vector<Line>().swap(lines);
	return *result;
}


/**
 * 判断两条直线是否具备聚合条件
 * 判断规则：斜率相近，直线间距相近，则可以聚合
 */
bool canCluster(Line l1, Line l2) {
	double th = (l1.length + l2.length) / 2.0;
	return abs(l1.k - l2.k) <= 0.3 &&  // 斜率差的绝对值小于0.5
		((l1.k > 0 && l2.k > 0) || (l1.k < 0 && l2.k < 0)) &&  // 斜率同号
		distanceBetweenLine(l1, l2) < th; // 距离较近
}

/**
 * 利用点到直线距离，计算两条直线的距离
 */
double distanceBetweenLine(Line l1, Line l2) {
	Point mid = l1.mid;
	double A = l2.k, B = -1, C = -(l2.k * l2.start.x - l2.start.y);
	return abs(A * mid.x + B * mid.y + C) / sqrt(A * A + B * B);
}

/**
 * 判断两个点是否相近
 * x，y间距均小于阈值
 */
bool isPointNear(Point p1, Point p2, double th){
	return (abs(p1.x - p2.x) < th && abs(p1.y - p2.y) < th);
}


/**
 * 判断直线首尾是否相接，并返回相连的类型
 * 0：不相连
 * 1：l1的end   和 l2的start   相连
 * 2：l1的end   和 l2的end     相连
 * 3：l1的start 和 l2的start   相连
 * 4：l1的start 和 l2的end     相连
*/
int isConnect(Line l1, Line l2, int th) {
	double len = max(l1.length, l2.length);
	int status = 0;
	if (isPointNear(l1.end, l2.start, th) && !isPointNear(l1.start, l2.end, len)) {
		status = 1;
	}else if (isPointNear(l1.end, l2.end, th) && !isPointNear(l1.start, l2.start, len)) {
		status = 2;
	}else if (isPointNear(l1.start, l2.start, th) && !isPointNear(l1.end, l2.end, len)) {
		status = 3;
	} else if (isPointNear(l1.start, l2.end, th) && !isPointNear(l1.end, l2.start, len)) {
		status = 4;
	}
	return status;
}


/**
 * 产生首尾相连的长直线
 */
Line createConnectLine(Line l1, Line l2, int type) {	
	Line l = Line();
	if (type == 1) {
		l.start = l1.start;
		l.end = l2.end;
	}else if (type == 2) {
		l.start = l1.start;
		l.end = l2.start;
	}else if (type == 3) {
		l.start = l1.end;
		l.end = l2.end;
	}else if (type == 4) {
		l.start = l1.end;
		l.end = l2.start;
	}
	l.length = l.getLength();
	l.mid = l.getMidPoint();	
	return l;
}


/**
 * 直线连接，如果两个直线首尾能相接
 * 则连接成一条直线
 * 第三个参数的作用就是测试的时候画图
 */
vector<Line> connectLines(vector<Line> lines, int th, Mat dst) {

	vector<Line> *result = new vector<Line>();
	size_t length = lines.size();

	for (int i = 0; i < length; i++) {
		Line line1 = lines[i];
		bool useless = true;
		for (int j = 0; j < length; j++) {
			Line line2 = lines[j];
			if (canCluster(line1, line2)) { // 如果具备聚合条件
				int type = isConnect(line1, line2, th); // 计算类型
                if (type != 0) {  // 如果是连接型
					useless = false;
					Line tmp = createConnectLine(line1, line2, type);
					(*result).push_back(tmp);
					line(dst, tmp.start, tmp.end, Scalar(0, 255, 0), 2, CV_AA);
					break;
				}
			}
		}
		if (useless) {
			(*result).push_back(line1);
		}
	}
	vector<Line>().swap(lines);
	return *result;
}


/**
 * 直线聚合函数,聚合的原则：
 * 如果两个直线的起点和终点相似，则保留那条长直线
 * 由于直线数量不多，采用暴力求解的方法，时间复杂度O(n2)
*/
vector<Line> clusterLines(vector<Line> lines, int th, Mat dst) {
	vector<Line> *result = new vector<Line>();
	size_t length = lines.size();
	
	// TODO 写一个hashmap

	for (int i = 0; i < length; i++) {
		Line line1 = lines[i];
		bool useless = true;
		for (int j = i; j < length; j++) {
			Line line2 = lines[j];
			
			if (canCluster(line1, line2)) { // 如果具备聚合条件
				useless = false;
				if (line1.length >= line2.length) {
					(*result).push_back(line1);
					line( dst, line1.start, line1.end, Scalar(255, 0, 0), 2, CV_AA);	
				}else {
					(*result).push_back(line2);
					line( dst, line2.start, line2.end, Scalar(255, 0, 0), 2, CV_AA);	
				}
				break;
			}
		}
	    if (useless) {
			(*result).push_back(line1);
		}
		
	}

	vector<Line>().swap(lines);
	return *result;
}




/*
计算两组直线的匹配度
输入：两个图像的两组直线 lines1，lines2
算法步骤如下：
1. 计算每组直线的斜率，计算斜率阈值TK、距离阈值TP
2. 根据斜率、距离的差值是否满足阈值，找到最佳匹配直线对
3. 计算每组中的直线与本组中的其他直线之间的夹角
4. 计算夹角矩阵之间的相似度，并把这个相似度，作为直线的匹配度，返回

TODO:
1. 直线匹配时，存在一些短直线相互之间离得很近，并且方向角度相似，其实是一条直线，应当聚类
2. 通过直线构造三种直线组合，利用直线组合还原高级特征，通过高级特征图匹配
*/
double match(vector<Vec4f> lines1, vector<Vec4f> lines2, InputArray m1, InputArray m2) {

	// Step1 创建直线
	vector<Line> lineSet1 = createLine(lines1);
	vector<Line> lineSet2 = createLine(lines2);

	vector<Vec4f>().swap(lines1);
	vector<Vec4f>().swap(lines2);

	int threshold = 5;
	Mat dst1(m1.getMat().rows, m1.getMat().cols, CV_8UC3, Scalar(255, 255, 255));
	Mat dst2(m1.getMat().rows, m1.getMat().cols, CV_8UC3, Scalar(255, 255, 255));

	// Step2 删除较短的直线
	lineSet1 = cleanShort(lineSet1);

	// Step3 先进行直线的连接
    lineSet1 = connectLines(lineSet1, threshold, dst1);

	// Step4 再进行直线的聚合
	lineSet1 = clusterLines(lineSet1, threshold, dst1); 
	//lineSet2 = clusterLines(lineSet2, threshold, dst2);

	size_t length1 = lineSet1.size();
	size_t length2 = lineSet2.size();


	line(dst1, Point(0, 0), Point(10, 0), Scalar(255, 0, 0), 3, CV_AA);

	// 画出聚合后的图像
	for (int i = 0; i < lineSet1.size(); i++) {
		Line l = lineSet1[i];
		line(dst1, l.start, l.end, Scalar(0, 0, 0), 1, CV_AA);
	}
	imshow("直线聚合后的图像1", dst1);
	/*for (int i = 0; i < lineSet2.size(); i++) {
		Line l = lineSet2[i];
		line(dst2, l.start, l.end, Scalar(0, 0, 0), 1, CV_AA);
	}
	imshow("直线聚合后的图像2", dst2);*/

	return 0.0;
}
