#include <opencv2/opencv.hpp>
#include <iostream>
#include <time.h>
#include <set>
#include "Line.h"

using namespace cv;
using namespace std;

vector<Vec4f> operation(string path, Mat image); // �������ͼ�����ֱ�߼��
vector<Line> createLine(vector<Vec4f> lines); // ����ֱ��
vector<Line> clusterLines(vector<Line> lines, int th, Mat dst); // ֱ�߾ۺ�
bool canCluster(Line l1, Line l2);
double distanceBetweenLine(Line l1, Line l2);
bool isPointNear(Point p1, Point p2, double th);
int isConnect(Line l1, Line l2, int th);
vector<Line> clusterLines(vector<Line> lines, int th, Mat dst);
double match(vector<Vec4f> lines1, vector<Vec4f> lines2,
	InputArray m1, InputArray m2); // ��������ֱ�ߵ�ƥ���


int main() {

	// Step1 ��ȡֱ��
	string path1 = "images/test3.jpg";
	string path2 = "images/test4.jpg";

	Mat image1 = imread(path1, IMREAD_GRAYSCALE);
	Mat image2 = imread(path2, IMREAD_GRAYSCALE);
	
	vector<Vec4f> lines_std1 = operation(path1, image1);
	vector<Vec4f> lines_std2 = operation(path2, image2);



	// Step2 �����������ȫ��֪��զ����ܳ
	// ֱ�ߵľ��࣬�ӳ��������ֱ��ˮƽ�����

	double rate = match(lines_std1, lines_std2, image1, image2);
	cout << "ƥ����ǣ�" << rate << endl;


	// Step3 �������������������ƥ���
	
	//path1 = "images/test3_c.jpg";
	//path2 = "images/test4_c.jpg";

	//Mat image1 = imread(path1, IMREAD_GRAYSCALE);
	//Mat image2 = imread(path2, IMREAD_GRAYSCALE);

	//image1 = 255 - image1; // ��ɫ
	//image2 = 255 - image2;

	//imshow("1", image1);
	//imshow("2", image2);
	//
	//Mat image1_copy = imread(path1);
	//Mat image2_copy = imread(path2);

	//vector<vector<Point>> contours1, contours2;
	//findContours(image1, contours1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);//���������  
	//findContours(image2, contours2, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);//���������

	//drawContours(image1_copy, contours1, -1, Scalar(0, 255, 0), 2, 8);
	//drawContours(image2_copy, contours2, -1, Scalar(0, 255, 0), 2, 8);

	//imshow("����1", image1_copy);
	//imshow("����2", image2_copy);

	////���ش�������ģ������֮������ƶ�,a0ԽСԽ����  
	//double a0 = matchShapes(contours1[0], contours2[0], CV_CONTOURS_MATCH_I1, 0);
	//cout << a0 << endl;

	waitKey(0);
	return 0;
}



/**
 * ͼ���˲�����Ե��⡢ֱ�߼�����
 * ���ؼ����ֱ��
 */
vector<Vec4f> operation(string path, Mat image) {
	blur(image, image, Size(3, 3)); // ʹ��3x3�ں�������
	Canny(image, image, 50, 200, 3); // Apply canny edge

	// Create and LSD detector with standard or no refinement.
	// LSD_REFINE_NONE��û�и����ķ�ʽ��
	// LSD_REFINE_STD����׼������ʽ���������ȵ��ߣ����ߣ���ɶ�����Աƽ�ԭ�߶ε�ֱ�߶ȣ�
	// LSD_REFINE_ADV����һ��������ʽ����������󾯸�������ͨ�����Ӿ��ȣ����ٳߴ��һ����ȷֱ�ߡ�
	Ptr<LineSegmentDetector> ls = createLineSegmentDetector(LSD_REFINE_STD);
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
 * �������е�ÿһ��ֱ�߹���Line����
 * ����һ���������ϣ��������Ԫ����Line����
 */
vector<Line> createLine(vector<Vec4f> lines) {
	vector<Line> LineSet;
	for (int i = 0; i < lines.size(); i++) {
		Line *line = new Line(lines[i]);
		LineSet.push_back(*line);
	}
	return LineSet;
}



/**
 * �ж�����ֱ���Ƿ�߱��ۺ�����
 * �жϹ���б�������ֱ�߼�����������Ծۺ�
 */
bool canCluster(Line l1, Line l2) {
	double th = (l1.length + l2.length) / 2.0;
	return abs(l1.k - l2.k) <= 1 && distanceBetweenLine(l1, l2) < th;
}

/**
 * ���õ㵽ֱ�߾��룬��������ֱ�ߵľ���
 */
double distanceBetweenLine(Line l1, Line l2) {
	Point mid = l1.mid;
	double A = l2.k, B = -1, C = -(l2.k * l2.start.x - l2.start.y);
	return abs(A * mid.x + B * mid.y + C) / sqrt(A * A + B * B);
}

/**
 * �ж��������Ƿ����
 * x��y����С����ֵ
 */
bool isPointNear(Point p1, Point p2, double th){
	return (abs(p1.x - p2.x) < th && abs(p1.y - p2.y) < th);
}


/**
 * �ж�ֱ����β�Ƿ���ӣ�����������������
 * -1��������
 * 0�����������Ͳ�����
 * 1��l1��end   �� l2��start   ����
 * 2��l1��end   �� l2��end     ����
 * 3��l1��start �� l2��start   ����
 * 4��l1��start �� l2��end     ����
*/
int isConnect(Line l1, Line l2, int th) {
	double len = max(l1.length, l2.length);
	int status = 0;
	if ((l1.k > 0 && l2.k < 0) || (l1.k < 0 && l2.k > 0) || abs(l1.k - l2.k) > 0.5) {
		status = -1;
	}else if (isPointNear(l1.end, l2.start, th) && !isPointNear(l1.start, l2.end, len)) {
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
 * ������β�����ĳ�ֱ��
 */
Line createConnectLine(Line l1, Line l2, int type) {
	Line l;
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
	return l;
}

/**
 * ֱ�߾ۺϺ���
 * ���Ⱦۺϵ�ԭ��
 *  - 1. �������ֱ�ߵ������յ����ƣ�����������ֱ��
 *  - 2. �������ֱ�ߵ���β����ӣ���ϲ���һ����ֱ��
 * ����ֱ���������࣬���ñ������ķ�����ʱ�临�Ӷ�O(n2)
*/
vector<Line> clusterLines(vector<Line> lines, int th, Mat dst) {
	vector<Line> *result = new vector<Line>();
	size_t length = lines.size();
	set<int> line_set;
	set<int>::iterator iter;

	// ���ж��ܷ�����
	for (int i = 0; i < length; i++) {
		Line line1 = lines[i];
		for (int j = 0; j < length; j++) {
			Line line2 = lines[j];
			if (canCluster(line1, line2)) { // ����߱��ۺ�����
				int type = isConnect(line1, line2, th); // ����ۺ�����
				if (type != 0 && type != -1) {  
					Line l = createConnectLine(line1, line2, type);
					(*result).push_back(l);
					//line( dst, l.start, l.end, Scalar(0, 255, 0), 3, CV_AA);				
					break;
				}

			}
		}
	}

	// Ȼ���ж��ܷ�ۺ�
	for (int i = 0; i < length; i++) {
		Line line1 = lines[i];
		for (int j = 0; j < length; j++) {
			Line line2 = lines[j];
			if (canCluster(line1, line2)) { // �ж�ֱ���Ƿ�߱��ۺ�����
				int type = isConnect(line1, line2, th);
				if (type == 0) {  // �ж��Ƿ���������
					if (line1.length > line2.length) {
						(*result).push_back(line1);
						//line( dst, line1.start, line1.end, Scalar(0, 0, 255), 2, CV_AA);	
					}
					else {
						(*result).push_back(line2);
						//line( dst, line2.start, line2.end, Scalar(0, 0, 255), 2, CV_AA);	
					}
					break;
				}

			}
		}
	}

	return *result;
}




/*
��������ֱ�ߵ�ƥ���
���룺����ͼ�������ֱ�� lines1��lines2
�㷨�������£�
1. ����ÿ��ֱ�ߵ�б�ʣ�����б����ֵTK��������ֵTP
2. ����б�ʡ�����Ĳ�ֵ�Ƿ�������ֵ���ҵ����ƥ��ֱ�߶�
3. ����ÿ���е�ֱ���뱾���е�����ֱ��֮��ļн�
4. ����нǾ���֮������ƶȣ�����������ƶȣ���Ϊֱ�ߵ�ƥ��ȣ�����

TODO:
1. ֱ��ƥ��ʱ������һЩ��ֱ���໥֮����úܽ������ҷ���Ƕ����ƣ���ʵ��һ��ֱ�ߣ�Ӧ������
2. ͨ��ֱ�߹�������ֱ����ϣ�����ֱ����ϻ�ԭ�߼�������ͨ���߼�����ͼƥ��
*/
double match(vector<Vec4f> lines1, vector<Vec4f> lines2, InputArray m1, InputArray m2) {

	// Step1 ����ֱ�߲��ۺ�
	vector<Line> lineSet1 = createLine(lines1);
	vector<Line> lineSet2 = createLine(lines2);

	int threshold = 5;
	Mat dst1(m1.getMat().rows, m1.getMat().cols, CV_8UC3, Scalar(255, 255, 255));
	Mat dst2(m1.getMat().rows, m1.getMat().cols, CV_8UC3, Scalar(255, 255, 255));

	// ֱ�߾ۺ�
	lineSet1 = clusterLines(lineSet1, threshold, dst1); 
	lineSet2 = clusterLines(lineSet2, threshold, dst2);

	size_t length1 = lineSet1.size();
	size_t length2 = lineSet2.size();

	// �����ۺϺ��ͼ��
	for (int i = 0; i < lineSet1.size(); i++) {
		Line l = lineSet1[i];
		//cout << "("<<l.start.x << "," << l.start.y << "), (" << l.end.x << "," << l.end.y << ")"<< endl;
		line(dst1, l.start, l.end, Scalar(0, 0, 0), 1, CV_AA);
	}
	imshow("ֱ�߾ۺϺ��ͼ��1", dst1);
	for (int i = 0; i < lineSet2.size(); i++) {
		Line l = lineSet2[i];
		//cout << "("<<l.start.x << "," << l.start.y << "), (" << l.end.x << "," << l.end.y << ")"<< endl;
		line(dst2, l.start, l.end, Scalar(0, 0, 0), 1, CV_AA);
	}
	imshow("ֱ�߾ۺϺ��ͼ��2", dst2);

	return 0.0;
}
