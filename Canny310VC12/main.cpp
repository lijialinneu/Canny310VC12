#include <opencv2/opencv.hpp>
#include <iostream>
#include <time.h>
#include <hash_set>
#include "Line.h"
#include"MinHeap.h"

using namespace cv;
using namespace std;


/***************************** ������������ start ***********************************/

vector<Vec4f> operation(string path, Mat image); // �������ͼ�����ֱ�߼��
vector<Line> createLine(vector<Vec4f> lines); // ����ֱ��

bool canCluster(Line l1, Line l2);            // �ж��ܷ�ۺϻ�����
double distanceBetweenLine(Line l1, Line l2); // ��������ֱ�߼�ľ���
bool isPointNear(Point p1, Point p2, double th); // �ж��������Ƿ�ӽ�
int isConnect(Line l1, Line l2, int th);      // �������ӵ�����

vector<Line> connectLines(vector<Line> lines, int th, Mat dst); // ����ֱ��
vector<Line> clusterLines(vector<Line> lines, int th, Mat dst); // �ۺ�ֱ��
vector<Line> getTopK(vector<Line> lines, int K); // ȡ��topK��ֱ��

double match(vector<Vec4f> lines1, vector<Vec4f> lines2, 
	InputArray m1, InputArray m2);     // ��������ֱ�ߵ�ƥ���
double averageK(vector<Line> LineSet); // ����ƽ��б�ʣ����ڼ���TK��
double getTP(InputArray m1, InputArray m2); // ����TP:������ֵ
double getAngle(double k1, double k2); // ��������ֱ�߼н�
double calculateMean(vector<vector<double>> m); // �����������ƶ�
double calculateCorr2(vector<vector<double>> m1,
	vector<vector<double>> m2);

/***************************** ������������ end *************************************/


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

	vector<Vec4f>().swap(lines_std1);
	vector<Vec4f>().swap(lines_std2);

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
 * �������е�ÿһ��ֱ�߹���Line����
 * ����һ���������ϣ��������Ԫ����Line����
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
 * ���˶�Сֱ��
 * TODO ���˹�����ֱ��
 */
vector<Line> cleanShort(vector<Line> lines) {
	vector<Line> *result = new vector<Line>();
	size_t length = lines.size();
	if (length == 0) return lines;

	// ����average����
	double sum = 0;
	for (int i = 0; i < length; i++) {
		Line line = lines[i];
		sum += line.length;
	}
	double avg = (sum / length) / 2;

	// ���˶�С��ֱ��
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
 * �ж�����ֱ���Ƿ�߱��ۺ�����
 * �жϹ���б�������ֱ�߼�����������Ծۺ�
 */
bool canCluster(Line l1, Line l2) {
	double th = (l1.length + l2.length) / 2.0;
	return abs(l1.k - l2.k) <= 0.3 &&  // б�ʲ�ľ���ֵС��0.5
		((l1.k > 0 && l2.k > 0) || (l1.k < 0 && l2.k < 0)) &&  // б��ͬ��
		distanceBetweenLine(l1, l2) < th; // ����Ͻ�
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
 * 0��������
 * 1��l1��end   �� l2��start   ����
 * 2��l1��end   �� l2��end     ����
 * 3��l1��start �� l2��start   ����
 * 4��l1��start �� l2��end     ����
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
 * ������β�����ĳ�ֱ��
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
 * ֱ�����ӣ��������ֱ����β�����
 * �����ӳ�һ��ֱ��
 * ���������������þ��ǲ��Ե�ʱ��ͼ
 */
vector<Line> connectLines(vector<Line> lines, int th, Mat dst) {

	vector<Line> *result = new vector<Line>();
	size_t length = lines.size();

	for (int i = 0; i < length; i++) {
		Line line1 = lines[i];
		bool useless = true;
		for (int j = 0; j < length; j++) {
			Line line2 = lines[j];
			if (canCluster(line1, line2)) { // ����߱��ۺ�����
				int type = isConnect(line1, line2, th); // ��������
                if (type != 0) {  // �����������
					useless = false;
					Line tmp = createConnectLine(line1, line2, type);
					(*result).push_back(tmp);
					//line(dst, tmp.start, tmp.end, Scalar(0, 0, 255), 3, CV_AA);
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
 * ֱ�߾ۺϺ���,�ۺϵ�ԭ��
 * �������ֱ�ߵ������յ����ƣ�����������ֱ��
 * ����ֱ���������࣬���ñ������ķ�����ʱ�临�Ӷ�O(n2)
 */
vector<Line> clusterLines(vector<Line> lines, int th, Mat dst) {
	vector<Line> *result = new vector<Line>();
	size_t length = lines.size();
	
	// TODO дһ��hashset����̬��ɾ���ۺϺ����õ�ֱ��
	hash_set<int> set;
	hash_set<int>::iterator pos;

	for (int i = 0; i < length; i++) {
		
		pos = set.find(i);
		if (pos != set.end()) { // �������
			continue;
		}
		Line line1 = lines[i];
		bool useless = true;
		for (int j = i; j < length; j++) {

			pos = set.find(j);
			if (pos != set.end()) { // �������
				continue;
			}
			Line line2 = lines[j];

			if (canCluster(line1, line2)) { // ����߱��ۺ�����
				set.insert(i);
				set.insert(j);

				useless = false;
				if (line1.length >= line2.length) {
					(*result).push_back(line1);
					//line(dst, line1.start, line1.end, Scalar(0, 255, 0), 2, CV_AA);	
				}else {
					(*result).push_back(line2);
					//line(dst, line2.start, line2.end, Scalar(0, 255, 0), 2, CV_AA);	
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


/**
 * ȡ��topK��ֱ��
 */
vector<Line> getTopK(vector<Line> lines, int K) {

	// MinHeap heap(K);
	MinHeap* heap = new MinHeap(K);

	// �����󶥶�
	(*heap).createMinHeap(lines);
	for (int i = K + 1; i< lines.size(); i++) {
		(*heap).insert(lines[i]);
	}
	// (*heap).print();
	lines = (*heap).getHeap();
	delete heap;
	return lines;
}


/**
 * ��������ֱ�ߵ�ƥ���
 * ���룺����ͼ�������ֱ�� lines1��lines2
 * �㷨�������£�
 * 1. ����ÿ��ֱ�ߵ�б�ʣ�����б����ֵTK��������ֵTP
 * 2. ����б�ʡ�����Ĳ�ֵ�Ƿ�������ֵ���ҵ����ƥ��ֱ�߶�
 * 3. ����ÿ���е�ֱ���뱾���е�����ֱ��֮��ļн�
 * 4. ����нǾ���֮������ƶȣ�����������ƶȣ���Ϊֱ�ߵ�ƥ��ȣ�����
 * TODO: ͨ��ֱ�߹�������ֱ����ϣ�����ֱ����ϻ�ԭ�߼�������ͨ���߼�����ͼƥ��
 */
double match(vector<Vec4f> lines1, vector<Vec4f> lines2, InputArray m1, InputArray m2) {

	// Step1 ����ֱ��
	vector<Line> lineSet1 = createLine(lines1);
	vector<Line> lineSet2 = createLine(lines2);

	vector<Vec4f>().swap(lines1);
	vector<Vec4f>().swap(lines2);

	int threshold = 7;
	Mat dst1(m1.getMat().rows, m1.getMat().cols, CV_8UC3, Scalar(255, 255, 255));
	Mat dst2(m1.getMat().rows, m1.getMat().cols, CV_8UC3, Scalar(255, 255, 255));

	// Step2 ɾ���϶̵�ֱ��
	// lineSet1 = cleanShort(lineSet1);
	// lineSet2 = cleanShort(lineSet2);

	// Step3 �Ƚ���ֱ�ߵ�����
    lineSet1 = connectLines(lineSet1, threshold, dst1);
	lineSet2 = connectLines(lineSet2, threshold, dst2);

	// Step4 �ٽ���ֱ�ߵľۺ�
	lineSet1 = clusterLines(lineSet1, threshold, dst1); 
	lineSet2 = clusterLines(lineSet2, threshold, dst2);

	size_t length1 = lineSet1.size();
	size_t length2 = lineSet2.size();

	// line(dst1, Point(0, 0), Point(10, 0), Scalar(255, 0, 0), 3, CV_AA);

	// �����ۺϺ��ͼ��
	for (int i = 0; i < lineSet1.size(); i++) {
		Line l = lineSet1[i];
		line(dst1, l.start, l.end, Scalar(0, 0, 0), 1, CV_AA);
	}
	imshow("���ӡ��ۺϺ��ͼ��1", dst1);
	for (int i = 0; i < lineSet2.size(); i++) {
		Line l = lineSet2[i];
		line(dst2, l.start, l.end, Scalar(0, 0, 0), 1, CV_AA);
	}
	imshow("���ӡ��ۺϺ��ͼ��2", dst2);

	// TODO ��ȡ���������ߣ�ˮƽ����ߣ���ֱ�����


	return 0.0;
}


/**
 * ����ƽ��б�ʣ����ڼ���TK��
 */
double averageK(vector<Line> LineSet) {
	double avg = 0;
	int count = 0;
	for (int i = 0; i < LineSet.size(); i++) {
		if (LineSet[i].k != 100000) {
			avg += LineSet[i].k;
			count++;
		}
	}
	avg /= count;
	return avg;
}


/*
 * ����TP:������ֵ
 */
double getTP(InputArray m1, InputArray m2) {
	return (m1.getMat().rows + m2.getMat().rows) / 6;
}


/**
 * ��������ֱ�߼н�
 */
double getAngle(double k1, double k2) {
	return atan(abs(k2 - k1) / (1 + k1 * k2));
}


/**
 * ����������������ƶ�
 * matlab�е�corr2()���������鷳
 */
double calculateMean(vector<vector<double>> m) {

	vector<double> *mean = new vector<double>();
	int p = 0;
	size_t len = m[p].size();
	for (size_t j = len - 1; j >= 0; j--) {
		double count = 0;
		for (size_t i = 0, k = j; i <= len - 1; k--, i++) {
			count += m[i][k];
		}
		count /= m.size();
		(*mean).push_back(count);
		p++;
	}

	double count = 0;
	len = (*mean).size();
	for (int i = 0; i < len; i++) {
		count += (*mean)[i];
	}
	count /= (len + 1);
	return count;
}


double calculateCorr2(vector<vector<double>> m1,
	vector<vector<double>> m2) {

	double mean1 = calculateMean(m1);
	double mean2 = calculateMean(m2);

	//�������
	double numerator = 0;
	size_t len = m1.size();
	for (size_t i = 0; i < len; i++) {
		size_t len1 = m1[i].size();
		for (size_t j = 0; j < len1; j++) {
			numerator += (m1[i][j] - mean1) * (m2[i][j] - mean2);
		}
		for (size_t j = len1; j <= len1; j++) {
			numerator += mean1 * mean2;
		}
	}

	//�����ĸ sqrt(pow(x,2) + pow(y,2));
	double d1 = 0;
	double d2 = 0;
	for (size_t i = 0; i < len; i++) {
		size_t len1 = m1[i].size();
		for (size_t j = 0; j < len1; j++) {
			d1 += pow((m1[i][j] - mean1), 2);
			d2 += pow((m2[i][j] - mean2), 2);
		}
		for (size_t j = len1; j <= len; j++) {
			d1 += pow(mean1, 2);
			d2 += pow(mean2, 2);
		}
	}
	double denominator = sqrt(d1) * sqrt(d2);

	if (numerator == 0) return 0.0;
	return numerator / denominator;
}
