#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include "Line.h"

using namespace cv;
using namespace std;

/**
 * С����, ���ڵ��ֵС�ں��ӵ�ֵ
 * ������Ҫ������һ�Ѻ����������ҳ�����ǰk�����ݣ�
 * ��õķ���������С������ֱ����ǰk�����ݽ���һ��С���ѣ�Ȼ�����ʣ�������
 * ��������� < �Ѷ�Ԫ��, ˵������k��������С������ҪС��ֱ������������������һ������
 * ��������� > �Ѷ��������򽫴����ͶѶ�����������Ȼ��ӶѶ����µ����ѣ�ʹ����������С���ѡ�
 */

class MinHeap
{
private:
	int maxsize; // �ѵĴ�С
	void filterDown(int begin); // ���µ�����
	vector<Line> arr;

public:
	MinHeap(int k);
	~MinHeap();

	void createMinHeap(vector<Line> lineSet); // ������
	void insert(Line val); // ����Ԫ��
	Line getTop();         // ��ȡ�Ѷ�Ԫ��
	//static double getLength(Line line); // ���㳤��
	vector<Line> getHeap(); // ��ȡ���е�ȫ��Ԫ��
};