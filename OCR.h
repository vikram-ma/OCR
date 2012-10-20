/*
 *  preprocessing.h
 *
 *
 *  Created by damiles on 18/11/08.
 *  Modified by Vikram OCR.h
 *
 */

#ifndef OCR_H_INCLUDED
#define OCR_H_INCLUDED


#ifdef _CH_
#pragma package <opencv>
#endif

#ifndef _EiC
#include <cv.h>
#include <highgui.h>
#include <ml.h>
#include <stdio.h>
#include <ctype.h>
#include <iostream>
#include <fstream>

#include "opencv2/imgproc/imgproc.hpp"

#endif
using namespace std;

class OCR{
	public:
		float classify(IplImage* img,int showResult);
		OCR (char* path, int classes, int samples);
		//void test();
	private:
        char file_path[255];
		int train_samples;
		int classes;
		CvMat* trainData;
		CvMat* trainClasses;
		int size;
		static const int K=1;
		CvKNearest *knn;
		void getData();
		void train();
		IplImage preprocessing(IplImage* imgSrc,int new_width, int new_height, int showResult = 0);
		CvRect findBB(IplImage* imgSrc);
		void findY(IplImage* imgSrc,int* min, int* max);
		void findX(IplImage* imgSrc,int* min, int* max);
		void test();
		void print(IplImage prs_image);
		void process(IplImage* imgSrc, int new_width, int new_height, int printResult, CvRect bb);
		void preprocessPara(IplImage* src, int new_width, int new_height, int printResult);
};
#endif //OCR_H_INCLUDED
