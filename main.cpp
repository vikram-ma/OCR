#ifdef _CH_
#pragma package <opencv>
#endif

#ifndef _EiC
#include <cv.h>
#include "OCR.h"
#endif

int main( int argc, char** argv )
{
    IplImage* imagev = cvLoadImage("./sampleUppercase.pbm", 0);
	//////////////////
	//My OCR
	//////////////////
	int* size = new int[1];
	OCR ocr("./OCR/", 52, 3);
	ocr.classify(imagev, 1, size);
    return 0;
}
