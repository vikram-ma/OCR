#ifdef _CH_
#pragma package <opencv>
#endif

#ifndef _EiC
#include <cv.h>
#include <iostream>
#include "OCR.h"
#endif

int main( int argc, char** argv )
{
    using std::cout;
    using std::endl;
    if(argc != 3)
    {
        cout <<"Usage <"<< argv[0]<<"> <path to directory containing samples> <test image>" << endl;
    }
    IplImage* imagev = cvLoadImage(argv[2], 0);
	//////////////////
	//My OCR
	//////////////////
	int* size = new int[1];
	OCR ocr(argv[1], 52, 3);
	ocr.classify(imagev, 1, size);
    return 0;
}
