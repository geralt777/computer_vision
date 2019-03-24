//Histogram_equalization, Filters and Laplacian pyramid blending

#include "./header.h"

using namespace std;
using namespace cv;

void help_message(char* argv[])
{
	
	cout << argv[0] << " 1 " << "[path to input image] " << "[output directory]" << endl;
	cout << argv[0] << " 2 " << "[path to input image1] " << "[path to input image2] " << "[output directory]" << endl;
	cout << argv[0] << " 3 " << "[path to input image1] " << "[path to input image2] " << "[output directory]" << endl;
}

// ===================================================
// ======== Histogram equalization =======
// ===================================================

Mat histogram_equalization(const Mat& img_in)
{
	Mat image = img_in; // Histogram equalization result
	int img_size = image.rows*image.cols;
	Mat new_image_hist(image.rows, image.cols, image.type());

	//equalizeHist(image, new_image_hist);

	//split the image into its 3 channels
	vector<Mat> img_channels;
	vector<Mat> op_img_channels;
	split(image, img_channels);

	int i;

	//equalizeHist(img_channels[0], img_channels[0]);
	//equalizeHist(img_channels[1], img_channels[1]);
	//equalizeHist(img_channels[2], img_channels[2]);
	int hist_size = 256;
	int histogram_0[256] = { 0 };
	int histogram_1[256] = { 0 };
	int histogram_2[256] = { 0 };

	for (int j = 0; j < img_channels[0].rows; j++)
	{
		for (int k = 0; k < img_channels[0].cols; k++)
		{
			histogram_0[(int)img_channels[0].at<uchar>(j, k)]++;
			histogram_1[(int)img_channels[1].at<uchar>(j, k)]++;
			histogram_2[(int)img_channels[2].at<uchar>(j, k)]++;
		}
	}

	float factor = 255.0 / img_size;

	int c_hist_0[256], c_hist_1[256], c_hist_2[256];
	c_hist_0[0] = histogram_0[0];
	c_hist_1[0] = histogram_1[0];
	c_hist_2[0] = histogram_2[0];
	for (i = 1; i < 256; i++)
	{
		c_hist_0[i] = histogram_0[i] + c_hist_0[i - 1];
		c_hist_1[i] = histogram_1[i] + c_hist_1[i - 1];
		c_hist_2[i] = histogram_2[i] + c_hist_2[i - 1];
	}

	int s_hist_0[256], s_hist_1[256], s_hist_2[256];
	for (i = 0; i < 256; i++)
	{
		s_hist_0[i] = cvRound((double)c_hist_0[i] * factor);
		s_hist_1[i] = cvRound((double)c_hist_1[i] * factor);
		s_hist_2[i] = cvRound((double)c_hist_2[i] * factor);
	}

	for (int j = 0; j < img_channels[0].rows; j++)
	{
		for (int k = 0; k < img_channels[0].cols; k++)
		{
			img_channels[0].at<uchar>(j, k) = saturate_cast<uchar>(s_hist_0[img_channels[0].at<uchar>(j, k)]);
			img_channels[1].at<uchar>(j, k) = saturate_cast<uchar>(s_hist_1[img_channels[1].at<uchar>(j, k)]);
			img_channels[2].at<uchar>(j, k) = saturate_cast<uchar>(s_hist_2[img_channels[2].at<uchar>(j, k)]);
		}
	}

	merge(img_channels, new_image_hist);

	return new_image_hist;
}

bool part1(char* argv[])
{
	// Read in input images
	Mat input_image = imread(argv[2], IMREAD_COLOR);

	// Histogram equalization
	Mat output_image = histogram_equalization(input_image);

	// Write out the result
	string output_name = string(argv[3]) + string("1.jpg");
	imwrite(output_name.c_str(), output_image);

	return true;
}

// ===================================================
// ===== Frequency domain filtering ======
// ===================================================

Mat low_pass_filter(const Mat& img_in)
{
	Mat I = img_in;

	Mat padded;
	int m = getOptimalDFTSize(I.rows);
	int n = getOptimalDFTSize(I.cols);
	copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);

	dft(complexI, complexI);

	Mat mylowfil(complexI.size(), CV_32F, Scalar(0.0));
	Mat tt(10, 10, CV_32F, Scalar(1.0));
	tt(Rect(0, 0, 10, 10)).copyTo(mylowfil(Rect(0, 0, 10, 10)));
	tt(Rect(0, 0, 10, 10)).copyTo(mylowfil(Rect(complexI.cols - 10, 0, 10, 10)));
	tt(Rect(0, 0, 10, 10)).copyTo(mylowfil(Rect(complexI.cols - 10, complexI.rows - 10, 10, 10)));
	tt(Rect(0, 0, 10, 10)).copyTo(mylowfil(Rect(0, complexI.rows - 10, 10, 10)));
	Mat c2;
	Mat padded1;
	int m1 = getOptimalDFTSize(mylowfil.rows);
	int n1 = getOptimalDFTSize(mylowfil.cols);
	copyMakeBorder(mylowfil, padded1, 0, m1 - mylowfil.rows, 0, n1 - mylowfil.cols, BORDER_CONSTANT, Scalar::all(0));
	Mat planes2[] = { Mat_<float>(padded1), Mat::zeros(padded1.size(), CV_32F) };
	merge(planes2, 2, c2);
	mulSpectrums(complexI, c2, complexI, DFT_COMPLEX_OUTPUT);

	Mat res;
	idft(complexI, res);
	Mat planes1[] = { Mat::zeros(complexI.size(), CV_32F), Mat::zeros(complexI.size(), CV_32F) };
	split(res, planes1);
	magnitude(planes1[0], planes1[1], res);
	normalize(res, res, 0, 1, NORM_MINMAX);
    res.convertTo(res, CV_8UC1, 255.0);
	return res;
}

Mat high_pass_filter(const Mat& img_in)
{
	Mat I = img_in;

	Mat padded;
	int m = getOptimalDFTSize(I.rows);
	int n = getOptimalDFTSize(I.cols);
	copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);

	dft(complexI, complexI);

	Mat myhighfil(complexI.size(), CV_32F, Scalar(1.0));
	Mat tt(10, 10, CV_32F, Scalar(0.0));
	tt(Rect(0, 0, 10, 10)).copyTo(myhighfil(Rect(0, 0, 10, 10)));
	tt(Rect(0, 0, 10, 10)).copyTo(myhighfil(Rect(complexI.cols - 10, 0, 10, 10)));
	tt(Rect(0, 0, 10, 10)).copyTo(myhighfil(Rect(complexI.cols - 10, complexI.rows - 10, 10, 10)));
	tt(Rect(0, 0, 10, 10)).copyTo(myhighfil(Rect(0, complexI.rows - 10, 10, 10)));
	Mat c2;
	Mat padded1;
	int m1 = getOptimalDFTSize(myhighfil.rows);
	int n1 = getOptimalDFTSize(myhighfil.cols);
	copyMakeBorder(myhighfil, padded1, 0, m1 - myhighfil.rows, 0, n1 - myhighfil.cols, BORDER_CONSTANT, Scalar::all(0));
	Mat planes2[] = { Mat_<float>(padded1), Mat::zeros(padded1.size(), CV_32F) };
	merge(planes2, 2, c2);
	mulSpectrums(complexI, c2, complexI, DFT_COMPLEX_OUTPUT);

	Mat res;
	idft(complexI, res);
	Mat planes1[] = { Mat::zeros(complexI.size(), CV_32F), Mat::zeros(complexI.size(), CV_32F) };
	split(res, planes1);
	magnitude(planes1[0], planes1[1], res);
	normalize(res, res, 0, 1, NORM_MINMAX);
    res.convertTo(res, CV_8UC1, 255.0);
	return res;
}

Mat deconvolution(const Mat& img_in)
{
	Mat img_out = img_in; // Deconvolution result

	return img_out;
}

bool part2(char* argv[])
{
	// Read in input images
	Mat input_image1 = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
	Mat input_image2 = imread(argv[3], IMREAD_COLOR);

	// Low and high pass filters
	Mat output_image1 = low_pass_filter(input_image1);
	Mat output_image2 = high_pass_filter(input_image1);

	// Deconvolution
	Mat output_image3 = deconvolution(input_image2);

	// Write out the result
	string output_name1 = string(argv[4]) + string("2.jpg");
	string output_name2 = string(argv[4]) + string("3.jpg");
	string output_name3 = string(argv[4]) + string("4.jpg");
	imwrite(output_name1.c_str(), output_image1);
	imwrite(output_name2.c_str(), output_image2);
	imwrite(output_name3.c_str(), output_image3);

	return true;
}

// ===================================================
// ===== Laplacian pyramid blending ======
// ===================================================

Mat laplacian_pyramid_blending(const Mat& img_in1, const Mat& img_in2)
{
	Mat img2, img1;
	Mat imgA = img_in1;
	Mat imgB = img_in2;
	img1 = Mat(imgA, Range::all(), Range(0, imgA.rows));
	img2 = Mat(imgB, Range(0, imgA.rows), Range(0, imgA.rows));

	Mat temp1 = img1.clone();
	vector<Mat> gpa;
	for (int i = 0; i < 5; i++)
	{
		pyrDown(temp1, temp1);
		gpa.push_back(temp1);
	}

	temp1 = img2.clone();
	vector<Mat> gpb;
	for (int i = 0; i < 5; i++)
	{
		pyrDown(temp1, temp1);
		gpb.push_back(temp1);
	}

	vector<Mat> lpa;
	lpa.push_back(gpa[4]);
	for (int i = 4; i >0; i--)
	{
		Mat temp2;
		pyrUp(gpa[i], temp2);
		Mat temp3;
		int r = gpa[i - 1].rows, c = gpa[i - 1].cols;
		if (gpa[i - 1].cols > temp2.cols)
			c = temp2.cols;
		if (gpa[i - 1].rows > temp2.rows)
			r = temp2.rows;
		Size w(c, r);
		resize(gpa[i - 1], gpa[i - 1], w);
		resize(temp2, temp2, w);
		subtract(gpa[i - 1], temp2, temp3);
		lpa.push_back(temp3);
	}

	vector<Mat> lpb;
	lpb.push_back(gpb[4]);
	for (int i = 4; i >0; i--)
	{
		Mat temp2;
		pyrUp(gpb[i], temp2);
		Mat temp3;
		int r = gpb[i - 1].rows, c = gpb[i - 1].cols;
		if (gpb[i - 1].cols > temp2.cols)
			c = temp2.cols;
		if (gpb[i - 1].rows > temp2.rows)
			r = temp2.rows;
		Size w(c, r);
		resize(gpb[i - 1], gpb[i - 1], w);
		resize(temp2, temp2, w);
		subtract(gpb[i - 1], temp2, temp3);
		lpb.push_back(temp3);
	}

	vector<Mat> ls;
	for (int i = 0; i < 5; i++)
	{
		Mat temp2 = lpa[i].clone();
		lpa[i](Rect(0, 0, lpa[i].cols / 2, lpa[i].rows)).copyTo(temp2(Rect(0, 0, lpa[i].cols / 2, lpa[i].rows)));
		lpb[i](Rect(lpa[i].cols / 2, 0, lpa[i].cols / 2, lpa[i].rows)).copyTo(temp2(Rect(lpa[i].cols / 2, 0, lpa[i].cols / 2, lpa[i].rows)));
		//ls[i] = temp2;
		ls.push_back(temp2);
		//std::memcpy(ls[i].data, temp2, rows*cols*sizeof(uchar));
	}

	Mat cur = ls[0];
	for (int i = 1; i < 5; i++)
	{
		pyrUp(cur, cur);
		int r = cur.rows, c = cur.cols;
		if (cur.cols > ls[i].cols)
			c = ls[i].cols;
		if (cur.rows > ls[i].rows)
			r = ls[i].rows;
		Size w(c, r);
		resize(cur, cur, w);
		resize(ls[i], ls[i], w);
		cv::add(cur, ls[i], cur);
	}
	Mat img_out = cur; // Blending result

	return img_out;
}

bool part3(char* argv[])
{
	Mat input_image1 = imread(argv[2], IMREAD_COLOR);
	Mat input_image2 = imread(argv[3], IMREAD_COLOR);

	// Histogram equalization
	Mat output_image = laplacian_pyramid_blending(input_image1, input_image2);

	// Write out the result
	string output_name = string(argv[4]) + string("5.jpg");
	imwrite(output_name.c_str(), output_image);

	return true;
}

int main(int argc, char* argv[])
{
	int part = -1;

	// Validate the input arguments
	if (argc < 4) {
		help_message(argv);
		exit(1);
	}
	else {
		part = atoi(argv[1]);

		if (part == 1 && !(argc == 4)) {
			help_message(argv);
			exit(1);
		}
		if (part == 2 && !(argc == 5)) {
			help_message(argv);
			exit(1);
		}
		if (part == 3 && !(argc == 5)) {
			help_message(argv);
			exit(1);
		}
		if (part > 3 || part < 1 || argc > 5) {
			cout << "Input parameters out of bound ..." << endl;
			exit(1);
		}
	}

	switch (part) {
	case 1: part1(argv); break;
	case 2: part2(argv); break;
	case 3: part3(argv); break;
	}

	return 0;
}
