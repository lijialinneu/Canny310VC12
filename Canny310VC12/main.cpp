/*************************************************

Copyright:bupt
Author: lijialin 1040591521@qq.com
Date:2017-10-14
Description:д��δ����ʱ��ֻ���ϵۺ���֪�����Ǹ����
            ����ֻ���ϵ�֪��

**************************************************/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <time.h>
#include <hash_set>
#include "Line.h"
#include "MinHeap.h"
#include "canny.h"

#define MAX 1.7976931348623158e+308

using namespace cv;
using namespace std;


/***************************** ������������ start ***********************************/

double solution(string path1, string path2);       // �������ڣ�����Ϊ����ͼƬ��·�� 
vector<Vec4f> operation(string path, Mat image); // �������ͼ�����ֱ�߼��
vector<Line> createLine(vector<Vec4f> lines);    // ����ֱ��
bool canCluster(Line l1, Line l2, int th);       // �ж��ܷ�ۺϻ�����
bool isPointNear(Point p1, Point p2, double th); // �ж��������Ƿ�ӽ�
double distanceBetweenLine(Line l1, Line l2);    // ��������ֱ�߼�ľ���
int isConnect(Line l1, Line l2, int th);         // �������ӵ�����

vector<Line> connectLines(vector<Line> lines, int th, Mat dst); // ����ֱ��
vector<Line> clusterLines(vector<Line> lines, int th, Mat dst); // �ۺ�ֱ��
vector<Line> getTopK(vector<Line> lines, int K);         // ȡ��topK��ֱ��
double lineDiff(vector<Line> line1, vector<Line> line2); // ��������ֱ�ߵĲ��
void drawLine(vector<Line> lineSet, Mat image, Scalar color, string name); // ��ͼ���л���ֱ��
double pointDistance(Point p1, Point p2);       // ����������ľ���
vector<vector<Line>> makePair(vector<Line> lineSet1, 
	vector<Line>lineSet2, int th);              // ����ֱ�߽������
double match(vector<Vec4f> lines1, vector<Vec4f> lines2, 
	InputArray m1, InputArray m2);              // ��������ֱ�ߵ�ƥ���
double getAngle(double k1, double k2);          // ��������ֱ�߼н�
double calculateMean(vector<vector<double>> m); // �����������ƶ�
double calculateCorr2(vector<vector<double>> m1,
	vector<vector<double>> m2);

/***************************** ������������ end *************************************/


int main() {

	string dir = "images/"; // ͼƬ��Ŀ¼

	/*

	// ����1����������ͼƬ������ƶ�
	cout << "��һ�����" << endl;
	string arr[] = { 
		"image0.jpg", "image0.jpg",
		"image0.jpg", "image1.jpg",
		"image2.jpg", "image3.jpg",
		"image4.jpg", "image5.jpg",
		"image6.jpg", "image7.jpg",
		"image8.jpg", "image9.jpg",
		"image10.jpg", "image11.jpg",
		"image12.jpg", "image13.jpg",
		"image14.jpg", "image15.jpg",
		"image16.jpg", "image17.jpg",
		"image18.jpg", "image19.jpg",
		"image20.jpg", "image21.jpg",
		"image22.jpg", "image23.jpg",
	}; // 25����������

    // �������в���
	for (int i = 0; i <= 24; i += 2) {
		string path1 = dir + arr[i];
		string path2 = dir + arr[i+1];
		double rate = solution(path1, path2);
	}
	cout << endl;


	// ����2�����۲�����ͼƬ������ƶ�
	cout << "�ڶ������" << endl;
	string arr1[] = {
		"image0.jpg", "image2.jpg",
		"image1.jpg", "image4.jpg",
		"image0.jpg", "image4.jpg",
		"image1.jpg", "image10.jpg",
		"image0.jpg", "image14.jpg",
		"image14.jpg", "image18.jpg",
		"image4.jpg", "image20.jpg",
		"image2.jpg", "image22.jpg",
	}; // 8����������

	for (int i = 0; i <= 7; i += 2) {
		string path1 = dir + arr1[i];
		string path2 = dir + arr1[i + 1];
		double rate = solution(path1, path2);
	}
	cout << endl;


	// ����3����ȫ�������
	cout << "���������" << endl;
	for (int i = 0; i <= 23; i++) {
		string path1 = dir + arr[i];
		int j = rand() % 23;
		string path2 = dir + arr[j];
		double rate = solution(path1, path2);
	}
	cout << endl;
	*/

	string path1 = dir + "image0.jpg";
	string path2 = dir + "image1.jpg";
	double rate = solution(path1, path2);

	
	waitKey(0);
	system("pause"); // ��ͣ
	return 0;
}


/**
 * ��������
 * ���룺����ͼƬ��·��string����
 * �����ƥ���double����
 */
double solution(string path1, string path2) {

	clock_t start, finish;// �����ʱ
	double totaltime;
	start = clock();


	// Step1 ��ȡֱ��
	Mat image1 = imread(path1, IMREAD_GRAYSCALE);
	Mat image2 = imread(path2, IMREAD_GRAYSCALE);

	// ����Ƿ���ͼƬ
	if (image1.empty()) {
		cout << "Cannot read image file: " <<  path1 << endl;
		return -1;
	}
	if (image2.empty()) {
		cout << "Cannot read image file: " << path2 << endl;
		return -1;
	}

	// ͼ���С�仯
	double height = (double)image1.rows / image1.cols * image2.cols;
	resize(image1, image1, Size(image2.cols, height), 0, 0, CV_INTER_LINEAR);

	vector<Vec4f> lines_std1 = operation(path1, image1);
	vector<Vec4f> lines_std2 = operation(path2, image2);


	// Step2 �����������ȫ��֪��զ����ܳ
	// ֱ�ߵľ��࣬�ӳ��������ֱ��ˮƽ�����

	double rate = match(lines_std1, lines_std2, image1, image2);

	vector<Vec4f>().swap(lines_std1);
	vector<Vec4f>().swap(lines_std2);

	finish = clock();
	totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
	cout << path1 << "��" << path2 << "�����ƶ���" << rate << " ��ʱ" << totaltime << "��" << endl;
	return rate;
}


/**
 * ͼ���˲�����Ե��⡢ֱ�߼�����
 * ���룺ͼƬ·����������������ͼƬ
 * ��������ؼ����ֱ��
 */
vector<Vec4f> operation(string path, Mat image) {
	blur(image, image, Size(3, 3));  // ʹ��3x3�ں�������
	// Canny(image, image, 50, 200, 3); // Apply canny edge
	
	myCanny(image, image, 50, 200, 3);
	// imwrite(path + "(50,200)��Ե�����ͼ.jpg", image);
	// imshow(path+"��Ե���", image);

	// Create and LSD detector with standard or no refinement.
	// LSD_REFINE_NONE��û�и����ķ�ʽ��
	// LSD_REFINE_STD����׼������ʽ���������ȵ��ߣ����ߣ���ɶ�����Աƽ�ԭ�߶ε�ֱ�߶ȣ�
	// LSD_REFINE_ADV����һ����������������󾯸����������ٳߴ��һ����ȷֱ�ߡ�
	Ptr<LineSegmentDetector> ls = createLineSegmentDetector(LSD_REFINE_STD, 0.99);
	vector<Vec4f> lines_std;

	// Detect the lines
	ls->detect(image, lines_std);

	
	// Show found lines
	//Mat drawnLines(image);
	Mat only_lines(image.size(), image.type());
	//ls->drawSegments(drawnLines, lines_std);
	ls->drawSegments(only_lines, lines_std);
	// imshow(path, drawnLines);
     imshow(path, only_lines);
	// imwrite(path + "ֱ����ȡ.jpg", only_lines);

	return lines_std;
}


/**
 * �������е�ÿһ��ֱ�߹���Line����
 * ����һ���������ϣ��������Ԫ����Line����
 * ���룺����Vec4f����ֱ�ߵ�����
 * ���������Line���������
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
 * ���룺����Line���������
 * ����������Сֱ�ߺ��Line��������
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
	double avg = sum / length;

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
 * ���룺����ֱ�ߣ�Line��������ֵ
 * �����bool�ͣ�true��ʾ����ֱ�߿��Ծۺϣ�false��ʾ���ܾۺ�
 */
bool canCluster(Line l1, Line l2, int th) {
	// double th = (l1.length + l2.length) / 2.0;
	return abs(l1.k - l2.k) <= 0.3 &&  // б�ʲ�ľ���ֵС��0.3
		((l1.k > 0 && l2.k > 0) || (l1.k < 0 && l2.k < 0)) &&  // б��ͬ��
		distanceBetweenLine(l1, l2) < th; // ����Ͻ�
}


/**
 * ����������ľ���
 * ���룺�����㣨Point���͵Ķ���
 * �����������֮��ľ��룬double����
 */
double pointDistance(Point p1, Point p2) {
	return abs(p1.x - p2.x) + abs(p1.y - p2.y);
}


/**
 * ԭ���������õ㵽ֱ�߾��룬��������ֱ�ߵľ���
 * �Ľ�������ֱ�߼��е�ľ���
 * ���룺����ֱ�ߣ�Line����
 * �����ֱ��֮��ľ��룬 double����
 */
double distanceBetweenLine(Line l1, Line l2) {
	/*Point mid = l1.mid;
	double A = l2.k, B = -1, C = -(l2.k * l2.start.x - l2.start.y);
	return abs(A * mid.x + B * mid.y + C) / sqrt(A * A + B * B);*/
	return pointDistance(l1.mid, l2.mid);
}


/**
 * �ж��������Ƿ����
 * x��y����С����ֵ
 * ���룺�����㣨Point���󣩣���ֵdouble����
 * �����bool���ͣ�true��ʾ��������úܽ���false��ʾ���Զ
 */
bool isPointNear(Point p1, Point p2, double th){
	return (abs(p1.x - p2.x) < th && abs(p1.y - p2.y) < th);
}


/**
 * �ж�ֱ����β�Ƿ���ӣ�����������������
 * ���룺����ֱ�ߣ�Line���󣩣���ֵ int
 * �����һ��int�ͣ���ʾ���������£�
 *    0��������
 *    1��l1��end   �� l2��start   ����
 *    2��l1��end   �� l2��end     ����
 *    3��l1��start �� l2��start   ����
 *    4��l1��start �� l2��end     ����
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
 * ���룺����ֱ�ߣ�Line���󣩣����� int
 * �����һ�����Ӻ��ֱ�ߣ�Line����
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
 * ֱ�����ӣ��������ֱ����β����ӣ������ӳ�һ��ֱ��
 * ���룺����Line�������������ֵ������Mat�����þ��ǲ��Ե�ʱ��ͼ�ã�
 * ���������Line���������
 */
vector<Line> connectLines(vector<Line> lines, int th, Mat dst) {
	vector<Line> *result = new vector<Line>();
	size_t length = lines.size();

	for (int i = 0; i < length; i++) {
		Line line1 = lines[i];
		bool useless = true;
		for (int j = 0; j < length; j++) {
			Line line2 = lines[j];
			if (canCluster(line1, line2, th)) { // ����߱��ۺ�����
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
 * ���룺����Line�������������ֵint������Mat�����þ��ǲ��Ե�ʱ��ͼ�ã�
 * ���������Line���������
 */
vector<Line> clusterLines(vector<Line> lines, int th, Mat dst) {
	vector<Line> *result = new vector<Line>();
	size_t length = lines.size();
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
			if (canCluster(line1, line2, th)) { // ����߱��ۺ�����
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
 * ������ȡ��topK��ֱ�ߣ����������ʱ�ò�����
 * ���룺����Line�����������TopK��K
 * ���������Line�������������ʾǰK��ֱ��
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
 * ��������ֱ�ߵĲ�࣬�㷨���£�
 *  - ���㳤�Ȳ��
 *  - ����б�ʵĲ��
 * ���أ����Ȳ�� * б�ʲ��
 * ���룺����ֱ�ߣ�Line����
 * �����ֱ��֮��Ĳ�࣬double����
 */
double lineDiff(Line line1, Line line2) {
	return abs(line1.length - line2.length) * abs(line1.k - line2.k);
}


/**
 * ��ͼ���л���ֱ�ߣ�������ʱ��
 * ���룺����Line�������������ʾֱ���飻Mat������ɫScalar���ͣ���������
 * �����void
 */
void drawLine(vector<Line> lineSet, Mat image, Scalar color, string name) {
	size_t len = lineSet.size();
	for (int i = 0; i < len; i++) {
		Line l = lineSet[i];
		line(image, l.start, l.end, color, 1, CV_AA);
	}
	imshow(name, image);
}


/**
 * ����ֱ�����е�ֱ�ߣ�����������ԣ�����˵ƥ��
 * ���룺����ֱ�ߣ�����Line�������������ֵ int
 * �����һ��n*2�Ķ�ά������n��ʾƥ��ĶԶ���
 */
vector<vector<Line>> makePair(vector<Line> lineSet1, vector<Line>lineSet2, int th) {
	hash_set<int> set;
	hash_set<int>::iterator pos;
	vector<vector<Line>> pairSet;
	size_t length1 = lineSet1.size();
	size_t length2 = lineSet2.size();

	for (int i = 0; i < length1; i++) {
		Line line1 = lineSet1[i];
		int bestFriendId = -1; // ������ֱ�ߵ�id
		double minDiff = MAX;  // ����ֱ����С�Ĳ��
		for (int j = 0; j < length2; j++) {
			pos = set.find(j);
			if (pos != set.end()) { // �������
				continue;
			}
			Line line2 = lineSet2[j];
			// ���ֱ��1,2��б����������������λ������������
			// ����͵������ֱ����canCluster()�����жϣ�����ܾۺϣ�Ҳ�������
			if (canCluster(line1, line2, th * 2)) {
				double diff = lineDiff(line1, line2);
				if (diff < minDiff) {
					minDiff = diff;
					bestFriendId = j;
				}
			}
		}
		// �ҵ������Ե�ֱ�ߺ󣬴洢����ά������
		if (bestFriendId != -1) {
			set.insert(bestFriendId);
			vector<Line> pair;
			Line bestFriendLine = lineSet2[bestFriendId];
			pair.push_back(line1);
			pair.push_back(bestFriendLine);
			pairSet.push_back(pair);
		}
	}
	vector<Line>().swap(lineSet1); // �����ڴ�
	vector<Line>().swap(lineSet2);
	return pairSet;
}


/**
 * ��������ֱ�ߵ�ƥ���
 * ���룺����ͼ�������ֱ�� lines1��lines2
 * �����ƥ��ȣ�double����
 *
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
	
	vector<Vec4f>().swap(lines1); // �����ڴ�
	vector<Vec4f>().swap(lines2);

	int threshold = 8; // ��ֵ��5-10��
	Mat dst1(m1.getMat().rows, m1.getMat().cols, CV_8UC3, Scalar(255,255,255));
	Mat dst2(m1.getMat().rows, m1.getMat().cols, CV_8UC3, Scalar(255,255,255));

	// Step2 ɾ���϶̵�ֱ�� (��ѡ)
	lineSet1 = cleanShort(lineSet1);
	lineSet2 = cleanShort(lineSet2);

	// Step3 �Ƚ���ֱ�ߵ����ӣ�Ȼ��ۺ�ֱ��
    lineSet1 = connectLines(lineSet1, threshold, dst1); // ����
	lineSet2 = connectLines(lineSet2, threshold, dst2);

	lineSet1 = clusterLines(lineSet1, threshold, dst1); // �ۺ�
	lineSet2 = clusterLines(lineSet2, threshold, dst2);
	
	size_t length1 = lineSet1.size();
	size_t length2 = lineSet2.size();

	// �����ۺϺ��ͼ�񣬱��ڷ���
	// line(dst1, Point(0, 0), Point(10, 0), Scalar(255, 0, 0), 3, CV_AA); // ������ֵ
	//drawLine(lineSet1, dst1, Scalar(0,0,0), "���ӡ��ۺϺ��ͼ��1");
	//drawLine(lineSet2, dst2, Scalar(0,0,0), "���ӡ��ۺϺ��ͼ��2");


	// Step4. �ӵ�һ��ͼ��ѡ��һ��ֱ�ߣ�Ȼ������ڶ���ͼ���ҵ���ѵ����ֱ��
	vector<vector<Line>> pairSet = makePair(lineSet1, lineSet2, threshold);
	size_t pairLen = pairSet.size(); // �ж��ٶ�ֱ��
	
	// ������Ժ��ͼ�񣬱��ڷ���
	//Mat dst3(m1.getMat().rows, m1.getMat().cols, CV_8UC3, Scalar(255,255,255));
	//Mat dst4(m2.getMat().rows, m2.getMat().cols, CV_8UC3, Scalar(255,255,255));	
	//for (int i = 0; i < pairLen; i++) {
	//	int b = rand() % 255; //�������������
	//	int g = rand() % 255;
	//	int r = rand() % 255;
	//	vector<Line> v = pairSet[i];
	//	line(dst3, v[0].start, v[0].end, Scalar(b, g, r), 2, CV_AA);
	//	line(dst4, v[1].start, v[1].end, Scalar(b, g, r), 2, CV_AA);
	//}
	//imshow("��Ժ��ͼ��1", dst3);
	//imshow("��Ժ��ͼ��2", dst4);


	// Step 5. ����ֱ��������ֱ�ߵļнǣ�����нǾ���
	vector<vector<double>> angleList1, angleList2;
	for (int i = 0; i < pairLen; i++) {
		vector<Line> v1 = pairSet[i];
		vector<double> angle1, angle2;

		for (int j = i + 1; j < pairLen; j++) {
			vector<Line> v2 = pairSet[j];
			 angle1.push_back(getAngle(v1[0].k, v2[0].k));
			 angle2.push_back(getAngle(v1[1].k, v2[1].k));
		}
		angleList1.push_back(angle1);
		angleList2.push_back(angle2);

		vector<double>().swap(angle1); // �����ڴ�
		vector<double>().swap(angle2);
	}

	// Ȼ�����нǾ�������ƶ�
	double rate = calculateCorr2(angleList1, angleList2);

	if (length1 != length2) {
		rate *= (double) min(length1, length2) / max(length1, length2); 
	}
	return rate;
}


/**
 * ��������ֱ�߼н�
 * ���룺б��k1��k2��double����
 * ����������нǶȣ�double����
 */
double getAngle(double k1, double k2) {
	return atan(abs(k2 - k1) / (1 + k1 * k2));
}


/**
 * ����������������ƶ�
 * matlab�е�corr2()���������鷳
 * ���룺��ά����
 * �������ά�����ľ�ֵ
 */
double calculateMean(vector<vector<double>> m) {
	double sum = 0.0;
	int num = 0;
	size_t rows = m.size();
	for (int i = 0; i < rows; i++) {
		size_t cols = m[i].size();
		for (int j = 0; j < cols; j++) {
			sum += m[i][j];
			num++;
		}
	}
	return sum / num;
}


/**
 * corr2����������������������ϵ��
 * ����������һ����[0,1]�����ڣ���ֵԽ��˵�����ƶ�Խ��
 * ���룺������ά����
 * ��������ǵ����ϵ����double����
 */
double calculateCorr2(vector<vector<double>> m1, vector<vector<double>> m2) {

	double mean1 = calculateMean(m1);
	double mean2 = calculateMean(m2);

	// ����һ���жϣ����ƽ��ֵ���ƣ�����Ϊ��������ȫ��ͬ��ͼ��
	if (abs(mean1 - mean2) <= 1e-6 ) { return 1.0; }

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

