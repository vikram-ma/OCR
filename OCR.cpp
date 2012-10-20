/*
 *  basicOCR.c
 *
 *
 *  Created by damiles on 18/11/08.
 *  Copyright 2008 Damiles. GPL License
 *
 */

 /* Modified by Vikram.M.A */

#include "OCR.h"

void OCR::getData()
{
	IplImage* src_image;
	IplImage prs_image;
	CvMat row,data;
	char file[255];
	char dataFile[255];
	std::ifstream labelStream;
	std::ostringstream outStringStream;
	char ch;
	int i,j;

	for(i =0; i<classes; i++){ //26
	    sprintf(dataFile,"%s%d/data.txt",file_path, i);
	    labelStream.open(dataFile);
	    labelStream >> ch;
	    labelStream.close();
		for( j = 0; j< train_samples; j++){ //3

			//Load file
			if(j<10)
				sprintf(file,"%s%d/%d0%d.pbm",file_path, i, i, j);

			else
				sprintf(file,"%s%d/%d%d.pbm",file_path, i, i, j);
			src_image = cvLoadImage(file,0);
			if(!src_image){
				printf("Error: Cant load image %s\n", file);
				//exit(-1);
			}
			//process file
			prs_image = preprocessing(src_image, size, size);
			//Set class label
			cvGetRow(trainClasses, &row, i*train_samples + j);
			cvSet(&row, cvRealScalar(ch));
			//Set data
			cvGetRow(trainData, &row, i*train_samples + j);

			IplImage* img = cvCreateImage( cvSize( size, size ), IPL_DEPTH_32F, 1 );
			//convert 8 bits image to 32 float image
			cvConvertScale(&prs_image, img, 0.0039215, 0);

			cvGetSubRect(img, &data, cvRect(0,0, size,size));

			CvMat row_header, *row1;
			//convert data matrix sizexsize to vecor
			row1 = cvReshape( &data, &row_header, 0, 1 );
			cvCopy(row1, &row, NULL);
		}
	}
}

void OCR::train()
{
	knn=new CvKNearest( trainData, trainClasses, 0, false, K );
}

float OCR::classify(IplImage* img, int showResult)
{
	float result;
    preprocessPara(img, size, size, 1);
	return result;

}

void OCR::print(IplImage prs_image)
{
    float result;
    int showResult = 1;
    CvMat data;
    CvMat* nearest=cvCreateMat(1,K,CV_32FC1);
    //Set data
	IplImage* img32 = cvCreateImage( cvSize( size, size ), IPL_DEPTH_32F, 1 );
	cvConvertScale(&prs_image, img32, 0.0039215, 0);
	cvGetSubRect(img32, &data, cvRect(0,0, size,size));
	CvMat row_header, *row1;
	row1 = cvReshape( &data, &row_header, 0, 1 );

	result=knn->find_nearest(row1,K,0,0,nearest,0);
	char r = result;
	int accuracy=0;
	for(int i=0;i<K;i++){
		if( nearest->data.fl[i] == result)
                    accuracy++;
	}
	float pre=100*((float)accuracy/(float)K);
	if(showResult==1){
		printf("|\t%c\t| \t%.2f%%  \t| \t%d of %d \t",r,pre,accuracy,K);
	}
//	printf("others\t");
//	for(int i=0; i<K; i++)
//	{
//	    char c = nearest->data.fl[i];
//	    printf("%c", c);
//	}

	printf(" \n---------------------------------------------------------------\n");
}
void OCR::test(){
	IplImage* src_image;
	IplImage prs_image;
	CvMat row,data;
	char file[255];
	int i,j;
	int error=0;
	int testCount=0;
	for(i =0; i<classes; i++){
		for( j = 50; j< 50+train_samples; j++){

			sprintf(file,"%s%d/%d%d.pbm",file_path, i, i , j);
			src_image = cvLoadImage(file,0);
			if(!src_image){
				printf("Error: Cant load image %s\n", file);
				//exit(-1);
			}
			//process file
			prs_image = preprocessing(src_image, size, size);
			float r=classify(&prs_image,0);
			if((int)r!=i)
				error++;

			testCount++;
		}
	}
	float totalerror=100*(float)error/(float)testCount;
	printf("System Error: %.2f%%\n", totalerror);

}

OCR::OCR(char* path, int classe, int samples)
{
	sprintf(file_path, "%s", path);
	//file_path = path;
	train_samples = samples;
	classes = classe;
	size = 80;
	trainData = cvCreateMat(train_samples*classes, size*size, CV_32FC1);
	trainClasses = cvCreateMat(train_samples*classes, 1, CV_32FC1);

	//Get data (get images and process it)
	getData();

	//train
	train();
	//Test
	//test();

	printf(" ---------------------------------------------------------------\n");
	printf("|\tClass\t|\tPrecision\t|\tAccuracy\t|\n");
	printf(" ---------------------------------------------------------------\n");
}

/*****************************************************************
*
* Find the min box. The min box respect original aspect ratio image
* The image is a binary data and background is white.
*
*******************************************************************/
void OCR::findX(IplImage* imgSrc,int* min, int* max)
{

    //cout << "Finding X" << endl;
	int i;
	int minFound=0;
	CvMat data;
	CvScalar maxVal=cvRealScalar(imgSrc->height * 255);
	CvScalar val=cvRealScalar(0);
	//For each col sum, if sum < width*255 then we find the min
	//then continue to end to search the max, if sum< width*255 then is new max
	for (i=0; i< imgSrc->width; i++){
	    val = cvRealScalar(0);
		cvGetCol(imgSrc, &data, i);
		val= cvSum(&data);
		if(val.val[0] < maxVal.val[0])
		{
			*max= i;
			if(!minFound)
			{
				*min= i;
				minFound= 1;
			}

		}
	}
}

void OCR::findY(IplImage* imgSrc,int* min, int* max)
{
    //cout << "Finding Y" << endl;
	int i;
	int minFound=0;
	CvMat data;
	CvScalar maxVal=cvRealScalar(imgSrc->width * 255);
	CvScalar val=cvRealScalar(0);
	//For each col sum, if sum < width*255 then we find the min
	//then continue to end to search the max, if sum< width*255 then is new max
	for (i=0; i< imgSrc->height; i++)
	{
	    val = cvRealScalar(0);
		cvGetRow(imgSrc, &data, i);
		val= cvSum(&data);
		if(val.val[0] < maxVal.val[0])
		{
			*max=i;
			if(!minFound)
			{
				*min= i;
				minFound= 1;
			}
		}
	}
}

CvRect OCR::findBB(IplImage* imgSrc){
	CvRect aux;
	int xmin, xmax, ymin, ymax;
	xmin=xmax=ymin=ymax=0;

	findX(imgSrc, &xmin, &xmax);
	findY(imgSrc, &ymin, &ymax);
	aux=cvRect(xmin, ymin, xmax-xmin, ymax-ymin);

	return aux;
}

IplImage OCR::preprocessing(IplImage* imgSrc,int new_width, int new_height, int printResult)
{
	IplImage* result;
	IplImage* scaledResult;

	CvMat data;
	CvMat dataA;
	CvRect bb;//bounding box
	CvRect bba;//boundinb box maintain aspect ratio

	//Find bounding box
	bb=findBB(imgSrc);


	//Get bounding box data and no with aspect ratio, the x and y can be corrupted
	cvGetSubRect(imgSrc, &data, cvRect(bb.x, bb.y, bb.width, bb.height));

	//Create image with this data with width and height with aspect ratio 1
	//then we get highest size betwen width and height of our bounding box
	int size=(bb.width>bb.height)?bb.width:bb.height;
	result=cvCreateImage( cvSize( size, size ), 8, 1 );
	cvSet(result,CV_RGB(255,255,255),NULL);
	//Copy de data in center of image
	int x=(int)floor((float)(size-bb.width)/2.0f);
	int y=(int)floor((float)(size-bb.height)/2.0f);
	//TODO: here x and y can be replaced by 0!


	cvGetSubRect(result, &dataA, cvRect(x,y,bb.width, bb.height));

	cvCopy(&data, &dataA, NULL);
	//Scale result


	scaledResult=cvCreateImage( cvSize( new_width, new_height ), 8, 1 );
	cvResize(result, scaledResult, CV_INTER_NN);

	//Return processed data
	if(printResult == 1)
	{
	   print(*scaledResult);
	}

	return *scaledResult;

}

void OCR::preprocessPara(IplImage* imgSrc, int new_width, int new_height, int printResult)
{
	int minY, maxY;
    int i;
	int minYFound=0;

	CvMat data;
	CvScalar maxVal=cvRealScalar(imgSrc->width * 255);
	CvScalar val=cvRealScalar(0);
	//For each col sum, if sum < width*255 then we find the min
	//then continue to end to search the max, if sum< width*255 then is new max.
        for (i=0; i< imgSrc->height; i++)
        {
            cvGetRow(imgSrc, &data, i);
            val= cvSum(&data);
            if(val.val[0] < maxVal.val[0])
            { // some data is found!
                maxY = i;
                if(!minYFound)
                {
                    minY = i;
                    minYFound = 1;
                }
            }
            else if(minYFound == 1)
            {
                //some data was found previously, but current row 'i' doesn't have any data.
                //So process from row 'minY' till row maxY
                int j;
                int minX, maxX;
                int minXFound=0;
                //CvMat data;
                CvScalar maxValx=cvRealScalar((maxY - minY) * 255);
                CvScalar valx=cvRealScalar(0);
                //For each col sum, if sum < width*255 then we find the min
                //then continue to end to search the max, if sum< width*255 then is new max
                for (j=0; j< imgSrc->width - 1; j++)
                {
                    //cout<<"Current value of j "<< j << endl;
                    valx=cvRealScalar(0);
                    //instead of taking sum of entire column get sum of sub part of it.
                    cvGetSubRect(imgSrc,&data, cvRect(j,minY,1,maxY-minY));
                    //cvGetCol(imgSrc, &data, i);
                    valx= cvSum(&data);
                    if(valx.val[0] < maxValx.val[0])
                    { //Some data found
                        maxX= j;
                        if(!minXFound){
                            minX= j;
                            minXFound= 1;
                        }
                    }
                    else if(minXFound == 1)
                    {
                        int maxYp;

                        CvScalar maxValyS = cvRealScalar((maxX-minX)*255);
                        CvScalar valyS = cvRealScalar(0);
                        //cout<< "Column has been processed minX " << minX <<" maxX "<< maxX << endl;
                        // from minx to maxx and miny to maxy
                        for(int k=maxY-1; k >= minY; k--)
                        {
                            cvGetSubRect(imgSrc, &data, cvRect(minX, k, maxX-minX,1));
                            valyS = cvSum(&data);
                            if(valyS.val[0] < maxValyS.val[0])
                            {
                                maxYp = k+1;
                                break;
                            }
                        }
                        //Some data was found previosly but current column 'j' doesn't have any data.
                        // so from minY to maxY and minX to maxX is the bounding box of character!
                        process(imgSrc, new_width, new_height, printResult, cvRect(minX, minY, maxX-minX, maxYp-minY));

//	CvPoint pt1,pt2;
//	pt1.x = minX;
//	pt1.y = minY;
//	pt2.x = minX;
//	pt2.y = maxYp;
//	cvLine(imgSrc, pt1, pt2, CV_RGB(0, 0, 0));
//
//	pt1.x = maxX;
//	pt2.x = maxX;
//
//    cvLine(imgSrc, pt1, pt2, CV_RGB(0, 0, 0));
//
//    pt1.x = minX;
//    pt1.y = minY;
//    pt2.x = maxX;
//    pt2.y = minY;
//
//    cvLine(imgSrc, pt1, pt2, CV_RGB(0, 0, 0));
//
//    pt1.y = maxYp;
//    pt2.y = maxYp;
//    cvLine(imgSrc, pt1, pt2, CV_RGB(0, 0, 0));
//
//	cvNamedWindow("scaled result", CV_WINDOW_AUTOSIZE);
//    cvShowImage("scaled result",imgSrc);
//
//    cvWaitKey(0);
                        minXFound = 0;
                    }
                }

                minYFound = 0;
            }
        }
	//If exit from loop was because max height was reached, but minFound has been set, then process from minFound till height.
	//This will not happen in the ideal examples I take :)
	//}
	//Find bounding box
	//bb=findBB(imgSrc);


}

void OCR::process(IplImage* imgSrc, int new_width, int new_height, int printResult, CvRect bb)
{
    IplImage* result;
	IplImage* scaledResult;

	CvMat data;
	CvMat dataA;
	CvRect bba;//boundinb box maintain aspect ratio

	//Get bounding box data and no with aspect ratio, the x and y can be corrupted
	cvGetSubRect(imgSrc, &data, cvRect(bb.x, bb.y, bb.width, bb.height));
	//Create image with this data with width and height with aspect ratio 1
	//then we get highest size betwen width and height of our bounding box
	int size=(bb.width>bb.height)?bb.width:bb.height;
	result=cvCreateImage( cvSize( size, size ), 8, 1 );
	cvSet(result,CV_RGB(255,255,255),NULL);
	//Copy data to center of image
	int x=(int)floor((float)(size-bb.width)/2.0f);
	int y=(int)floor((float)(size-bb.height)/2.0f);
	//TODO: here x and y can be replaced by 0!
	cvGetSubRect(result, &dataA, cvRect(x,y,bb.width, bb.height));
	cvCopy(&data, &dataA, NULL);
	//Scale result
	scaledResult=cvCreateImage( cvSize( new_width, new_height ), 8, 1 );
	cvResize(result, scaledResult, CV_INTER_NN);
	//Return processed data
	if(printResult == 1)
	{
	   print(*scaledResult);
	}
}
