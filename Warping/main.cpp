#include "./header.h"

using namespace cv;
using namespace std;

void help_message(char* argv[])
{
	cout << "Usage: [Question_Number] [Input_Options] [Output_Options]" << endl;
	cout << "[Question Number]" << endl;
	cout << "1 Perspective warping" << endl;
	cout << "2 Cylindrical warping" << endl;
	cout << "3 Bonus perspective warping" << endl;
	cout << "4 Bonus cylindrical warping" << endl;
	cout << "[Input_Options]" << endl;
	cout << "Path to the input images" << endl;
	cout << "[Output_Options]" << endl;
	cout << "Output directory" << endl;
	cout << "Example usages:" << endl;
	cout << argv[0] << " 1 " << "[path to input image1] " << "[path to input image2] " << "[path to input image3] " << "[output directory]" << endl;
}

/*
Detect, extract and match features between img1 and img2.
Using SIFT as the detector/extractor, but this is inconsequential to the user.

Returns: (pts1, pts2), where ptsN are points on image N.
The lists are "aligned", i.e. point i in pts1 matches with point i in pts2.
*/
bool feature_matching(
	const Mat& img1,
	const Mat& img2,
	vector<KeyPoint>& good_keypoints_1,
	vector<KeyPoint>& good_keypoints_2,
	bool savefig = false
)
{
	// Check if the images are read in
	if (!img1.data || !img2.data) {
		cout << "Error reading images" << endl;
		exit(1);
	}

	// Initiate SIFT detector
	Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();

	// Find the keypoints
	vector<KeyPoint> keypoints_1, keypoints_2;
	f2d->detect(img1, keypoints_1);
	f2d->detect(img2, keypoints_2);

	// Calculate descriptors
	Mat descriptors_1, descriptors_2;
	f2d->compute(img1, keypoints_1, descriptors_1);
	f2d->compute(img2, keypoints_2, descriptors_2);

	// Matching descriptors using BFMatcher
	BFMatcher matcher(NORM_L2);
	vector<vector<DMatch> > matches;
	matcher.knnMatch(descriptors_1, descriptors_2, matches, 2);

	// Find good matches
	vector<DMatch> good_matches;
	for (int i = 0; i < matches.size(); i++) {
		assert(matches[i].size() == 2);
		if (matches[i][0].distance < 0.7 * matches[i][1].distance) {
			good_matches.push_back(matches[i][0]);
			good_keypoints_1.push_back(keypoints_1[matches[i][0].queryIdx]);
			good_keypoints_2.push_back(keypoints_2[matches[i][0].trainIdx]);
		}
	}

	// Show matching results as picture
	if (savefig) {
		Mat matching_res;
		drawMatches(img1, keypoints_1, img2, keypoints_2, good_matches, matching_res);
		imshow("Matching_results", matching_res);
		waitKey();
	}

	return true;
}

/*
Warp an image from cartesian coordinates (x, y) into cylindrical coordinates (theta, h)
Returns: (image, mask)
Mask is [0,255], and has 255s wherever the cylindrical images has a valid value.
Masks are useful for stitching
*/
bool cylindricalWarpImage(
	const Mat& img1,
	const Mat& K,
	Mat& cyl,
	Mat& cyl_mask,
	bool savefig = false)
{
	// Retrieve focal length, height and width
	float f = K.at<float>(0, 0);
	int height = img1.rows;
	int width = img1.cols;

	// Go inverse from cylindrical coordinates to the image
	// (this way there are no gaps)
	cyl = Mat(height, width, CV_8UC1, Scalar(0));
	cyl_mask = Mat(height, width, CV_32SC1, Scalar(0));

	float x_c = float(width) / 2.0;
	float y_c = float(height) / 2.0;
	for (int x_cyl = 0; x_cyl < width; x_cyl++) {
		for (int y_cyl = 0; y_cyl < height; y_cyl++) {
			float theta = (x_cyl - x_c) * 1.0f / f;
			float h = (y_cyl - y_c) * 1.0f / f;

			Mat X = (Mat_<float>(3, 1) << sin(theta), h, cos(theta));
			Mat dotProduct = K * X;

			float x_im = dotProduct.at<float>(0, 0) / dotProduct.at<float>(2, 0);
			if (x_im < 0 || x_im >= width) {
				continue;
			}

			float y_im = dotProduct.at<float>(1, 0) / dotProduct.at<float>(2, 0);
			if (y_im < 0 || y_im > height) {
				continue;
			}

			// Assign the values
			cyl.at<uchar>(y_cyl, x_cyl) = img1.at<uchar>(int(y_im), int(x_im));
			cyl_mask.at<int>(y_cyl, x_cyl) = 255;
		}
	}

	if (savefig) {
		imshow("Cylindrical warping", cyl);
		waitKey();
	}

	return true;
}

/*
Calculate the geometric transform (only affine or homography) between two images,
based on feature matching and alignment with a robust estimator (RANSAC).

Returns: (M, pts1, pts2, mask)
Where: M    is the 3x3 transform matrix
pts1 are the matched feature points in image 1
pts2 are the matched feature points in image 2
mask is a binary mask over the lists of points that selects the transformation inliers
*/
bool getTransform(
	const Mat& src,
	const Mat& dst,
	vector<Point2f>& pts1,
	vector<Point2f>& pts2,
	Mat& M,
	Mat& mask,
	string method = "affine"
)
{
	// Compute keypoints for the image pair
	vector<KeyPoint> keypoints_1, keypoints_2;
	feature_matching(src, dst, keypoints_1, keypoints_2);

	// Convert Keypoint vector to Point vector
	assert(keypoints_1.size() == keypoints_2.size());
	vector<Point2f> points1(keypoints_1.size());
	vector<Point2f> points2(keypoints_1.size());
	for (int i = 0; i < keypoints_1.size(); i++) {
		points1[i] = keypoints_1[i].pt;
		points2[i] = keypoints_2[i].pt;
	}

	// Compute transformation
	if (method == "affine") {
		M = estimateAffine2D(points1, points2, mask, RANSAC, 5.0);
		//Mat lastRow = (Mat_<double>(1, 3) << 0, 0, 1);
		//vconcat(M, lastRow, M);
	}

	if (method == "homography") {
		M = findHomography(points1, points2, RANSAC, 5.0, mask);
	}

	return true;
}

// ===================================================
// ================ Perspective Warping ==============
// ===================================================
bool Perspective_warping(char* argv[], const Mat& img1, const Mat& img2, const Mat& img3)
{

	//Copy first image to the final image
	Mat out(1312, 1608, img1.type());
	copyMakeBorder(img1, out, 200, 200, 500, 500, BORDER_CONSTANT);

	//Create the transformation of img 2 wrt img 1
	vector<Point2f> pts1, pts2;
	Mat tmat, mask;
	Mat temp1(1312, 1608, img1.type());

	getTransform(img2, out, pts1, pts2, tmat, mask, "homography");
	warpPerspective(img2, temp1, tmat, temp1.size());

	//Create the transformation of img 3 wrt img 1
	vector<Point2f> pts3, pts4;
	Mat tmat1, mask1;
	Mat temp2(1312, 1608, img1.type());

	getTransform(img3, out, pts3, pts4, tmat1, mask1, "homography");
	warpPerspective(img3, temp2, tmat1, temp2.size());

	//Make the panorama
	for (int i = 0; i < 1312; i++)
	{
		for (int j = 0; j < 1608; j++)
		{
			if (out.at<uchar>(i, j) == 0)
			{
				out.at<uchar>(i, j) = temp1.at<uchar>(i, j);
			}
		}
	}

	for (int i = 0; i < 1312; i++)
	{
		for (int j = 0; j < 1608; j++)
		{
			if (out.at<uchar>(i, j) == 0)
			{
				out.at<uchar>(i, j) = temp2.at<uchar>(i, j);
			}
		}
	}

	// Write out the result
	string output_name = string(argv[5]) + string("output_homography.png");
	imwrite(output_name.c_str(), out);

	return true;
}

bool Bonus_perspective_warping(char* argv[], const Mat& img1, const Mat& img2, const Mat& img3)
{
	Mat output_image = img1; 

							 
	string output_name = string(argv[5]) + string("output_homography_lpb.png");
	imwrite(output_name.c_str(), output_image);

	return true;
}

// ===================================================
// =============== Cynlindrical Warping ==============
// ===================================================
bool Cylindrical_warping(char* argv[], const Mat& img1, const Mat& img2, const Mat& img3)
{
	Mat K1(3, 3, CV_32FC1);
	K1.at<float>(0, 0) = 430; K1.at<float>(0, 1) = 0; K1.at<float>(0, 2) = (img1.cols) / 2;
	K1.at<float>(1, 0) = 0; K1.at<float>(1, 1) = 430; K1.at<float>(1, 2) = (img1.rows) / 2;
	K1.at<float>(2, 0) = 0; K1.at<float>(2, 1) = 0; K1.at<float>(2, 2) = 1;

	Mat K2(3, 3, CV_32FC1);
	K2.at<float>(0, 0) = 430; K2.at<float>(0, 1) = 0; K2.at<float>(0, 2) = (img2.cols) / 2;
	K2.at<float>(1, 0) = 0; K2.at<float>(1, 1) = 430; K2.at<float>(1, 2) = (img2.rows) / 2;
	K2.at<float>(2, 0) = 0; K2.at<float>(2, 1) = 0; K2.at<float>(2, 2) = 1;

	Mat K3(3, 3, CV_32FC1);
	K3.at<float>(0, 0) = 430; K3.at<float>(0, 1) = 0; K3.at<float>(0, 2) = (img3.cols) / 2;
	K3.at<float>(1, 0) = 0; K3.at<float>(1, 1) = 430; K3.at<float>(1, 2) = (img3.rows) / 2;
	K3.at<float>(2, 0) = 0; K3.at<float>(2, 1) = 0; K3.at<float>(2, 2) = 1;

	Mat cyl1, cyl2, cyl3, cyl_mask1, cyl_mask2, cyl_mask3;

	bool val1 = cylindricalWarpImage(img1, K1, cyl1, cyl_mask1);
	bool val2 = cylindricalWarpImage(img2, K2, cyl2, cyl_mask2);
	bool val3 = cylindricalWarpImage(img3, K3, cyl3, cyl_mask3);

	//Copy first image to the final image
	Mat out(1012, 1208, cyl1.type());
	copyMakeBorder(cyl1, out, 50, 50, 300, 300, BORDER_CONSTANT);

	//Create the transformation of img 2 wrt img 1
	vector<Point2f> pts1, pts2;
	Mat tmat, mask;
	Mat temp1(1012, 1208, cyl1.type());

	getTransform(cyl2, out, pts1, pts2, tmat, mask, "affine");
	warpAffine(cyl2, temp1, tmat, temp1.size());

	//Create the transformation of img 3 wrt img 1
	vector<Point2f> pts3, pts4;
	Mat tmat1, mask1;
	Mat temp2(1012, 1208, cyl1.type());

	getTransform(cyl3, out, pts3, pts4, tmat1, mask1, "affine");
	warpAffine(cyl3, temp2, tmat1, temp2.size());

	for (int i = 0; i < 1012; i++)
	{
		for (int j = 0; j < 1208; j++)
		{
			if (out.at<uchar>(i, j) == 0)
			{
				out.at<uchar>(i, j) = temp1.at<uchar>(i, j);
			}
		}
	}

	for (int i = 0; i < 1012; i++)
	{
		for (int j = 0; j < 1208; j++)
		{
			if (out.at<uchar>(i, j) == 0)
			{
				out.at<uchar>(i, j) = temp2.at<uchar>(i, j);
			}
		}
	}

	string output_name = string(argv[5]) + string("output_cylindrical.png");
	imwrite(output_name.c_str(), out);

	return true;
}


bool Bonus_cylindrical_warping(char* argv[], const Mat& img1, const Mat& img2, const Mat& img3)
{
	Mat output_image = img1;
	string output_name = string(argv[5]) + string("output_cylindrical_lpb.png");
	imwrite(output_name.c_str(), output_image);

	return true;
}

float RMSD(const int questionID, const Mat&  target, const Mat& master)
{
	// Get width, height, and number of channels of the master image
	int master_height = master.rows;
	int	master_width = master.cols;
	int master_channel = master.channels();

	// Get width, height, and number of channels of the target image
	int target_height = target.rows;
	int	target_width = target.cols;
	int target_channel = target.channels();

	// Validate the height, width and channels of the input image
	if (master_height != target_height || master_width != target_width || master_channel != target_channel) {
		return -1;
	}
	else {

		int nonZero_target = countNonZero(target);
		int nonZero_master = countNonZero(master);

		if (questionID == 1) {
			if (nonZero_target < 1200000) {
				return -1;
			}
		}
		else if (questionID == 2) {
			if (nonZero_target < 700000) {
				return -1;
			}
		}
		else {
			return -1;
		}

		float total_diff = 0.0;
		Mat master_channels[3];
		Mat target_channels[3];
		split(master, master_channels);
		split(target, target_channels);

		for (int i = 0; i < master_channel; i++) {
			Mat diff_mat;
			absdiff(master_channels[i], target_channels[i], diff_mat);
			pow(diff_mat, 2.0, diff_mat);
			float mean_value = mean(diff_mat)[0];
			total_diff = total_diff + sqrt(mean_value);
		}

		return total_diff;
	}
}

int main(int argc, char* argv[])
{
	int question_number = -1;

	// Validate the input arguments
	if (argc != 6) {
		help_message(argv);
		exit(1);
	}
	else {
		question_number = atoi(argv[1]);
		if (question_number > 4 || question_number < 1) {
			cout << "Input parameters out of bound ..." << endl;
			exit(1);
		}
	}

	Mat input_image1 = imread(argv[2], IMREAD_GRAYSCALE);
	Mat input_image2 = imread(argv[3], IMREAD_GRAYSCALE);
	Mat input_image3 = imread(argv[4], IMREAD_GRAYSCALE);

	switch (question_number) {
	case 1: Perspective_warping(argv, input_image1, input_image2, input_image3); break;
	case 2: Cylindrical_warping(argv, input_image1, input_image2, input_image3); break;
	case 3: Bonus_perspective_warping(argv, input_image1, input_image2, input_image3); break;
	case 4: Bonus_cylindrical_warping(argv, input_image1, input_image2, input_image3); break;
	}

	return 0;
}
