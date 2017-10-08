#include <opencv2/opencv.hpp>
#include <iostream>
#include <time.h>

using namespace cv;
using namespace std;

vector<Vec4f> operation(string path); // �������ͼ�����ֱ�߼��

int main() 
{
	string path1 = "images/test3.jpg";
	string path2 = "images/test4.jpg";

	vector<Vec4f> lines_std1 = operation(path1);
	vector<Vec4f> lines_std2 = operation(path2);

	// ֱ�ߵľ��࣬�ӳ��������ֱ��ˮƽ�����

	// �������������������ƥ���
	path1 = "images/test3_c.jpg";
	path2 = "images/test4_c.jpg";

	Mat image1 = imread(path1, IMREAD_GRAYSCALE);
	Mat image2 = imread(path2, IMREAD_GRAYSCALE);

	imshow("����1", image1);
	imshow("����2", image2);

	Moments m1, m2;

	vector<vector<Point>> contours1, contours2;
	findContours(image1, contours1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);//���������  
	findContours(image2, contours2, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);//���������

	//drawContours(copyImg1, contours1, -1, Scalar(0, 255, 0), 2, 8);
	//drawContours(copyImg1, contours1, -1, Scalar(0, 255, 0), 2, 8);

	//���ش�������ģ������֮������ƶ�,a0ԽСԽ����  
	double a0 = matchShapes(contours1[0], contours2[0], CV_CONTOURS_MATCH_I1, 0);
	cout << a0 << endl;

	waitKey(0);
	return 0;
}



/**
 * ͼ���˲�����Ե��⡢ֱ�߼�����
 * ���ؼ����ֱ��
 */
vector<Vec4f> operation(string path)
{
	Mat image = imread(path, IMREAD_GRAYSCALE);
	
	blur(image, image, Size(3, 3)); // ʹ��3x3�ں�������
	Canny(image, image, 50, 200, 3); // Apply canny edge

	// Create and LSD detector with standard or no refinement.
	// LSD_REFINE_NONE��û�и����ķ�ʽ��
	// LSD_REFINE_STD����׼������ʽ���������ȵ��ߣ����ߣ���ɶ�����Աƽ�ԭ�߶ε�ֱ�߶ȣ�
	// LSD_REFINE_ADV����һ��������ʽ����������󾯸�������ͨ�����Ӿ��ȣ����ٳߴ��һ����ȷֱ�ߡ�
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
