#include "MinHeap.h"
#include "Line.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>


using namespace std;
using namespace cv;


MinHeap::MinHeap(int k) {
	maxsize = k;
}

MinHeap::~MinHeap() {
	//minheap.clear();
	vector<Line>().swap(arr);
	//delete (*minheap); //�ͷſռ�
}


/**
 * ����С����
 */
void MinHeap::createMinHeap(vector<Line> lineSet) {
	for (int i = 0; i < maxsize; i++) {
		arr.push_back(lineSet[i]);
	}
}


/**
 * ����Ԫ��
 */
void MinHeap::insert(Line line) {

	if (line.length > getTop().length) {
		arr[0] = line;
		filterDown(0);
	}
}


/**
 * ���µ���
 */
void MinHeap::filterDown(int current) {
	int end = arr.size() - 1;
	int child = current * 2 + 1; //��ǰ�ڵ������
	Line line = arr[current]; //���浱ǰ�ڵ�

	while (child <= end) {
		// ѡ�����������еĽ�С����
		if (child < end && arr[child + 1].length < arr[child].length)
			child++;
		if (line.length < arr[child].length) break;
		else {
			arr[current] = arr[child];//���ӽڵ㸲�ǵ�ǰ�ڵ�
			current = child;
			child = child * 2 + 1;
		}
	}
	arr[current] = line;
}


/**
 * ��ȡ�Ѷ�Ԫ��
 */
Line MinHeap::getTop() {
	if (arr.size() != 0)
		return arr[0];
	return NULL;
}


/** 
 * ��ȡ���е�ȫ��Ԫ��
 */
vector<Line> MinHeap::getHeap() {
	vector<Line> heap;
	for (int i = 0; i < arr.size(); i++)
		heap.push_back(arr[i]);
	return heap;
}
