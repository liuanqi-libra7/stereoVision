#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>
#include "opencv2/opencv.hpp"
#include "opencv2/ximgproc.hpp"

using namespace cv;
using namespace std;
using namespace ximgproc;

bool get_calib(std::string intrinsic_filename, std::string extrinsic_filename, Size img_size, vector<Mat>& matrixs) {
	FileStorage fs(intrinsic_filename, FileStorage::READ);
	if (!fs.isOpened())
	{
		printf("Failed to open file %s\n", intrinsic_filename.c_str());
		return false;
	}

	Mat M1, D1, M2, D2;
	fs["M1"] >> M1;
	fs["D1"] >> D1;
	fs["M2"] >> M2;
	fs["D2"] >> D2;

	fs.open(extrinsic_filename, FileStorage::READ);
	if (!fs.isOpened())
	{
		printf("Failed to open file %s\n", extrinsic_filename.c_str());
		return false;
	}

	Mat R, T, R1, P1, R2, P2;
	fs["R"] >> R;
	fs["T"] >> T;

	Rect roi1, roi2;
	Mat Q;

	stereoRectify(M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2);

	Mat map11, map12, map21, map22;
	initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
	initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);
	matrixs.push_back(map11);
	matrixs.push_back(map12);
	matrixs.push_back(map21);
	matrixs.push_back(map22);
	matrixs.push_back(Q);

	return true;
}

int main(int argc, char** argv){

	std::string intrinsic_filename = "1280_720_intrinsics.yml";
	std::string extrinsic_filename = "1280_720_extrinsics.yml";
	Size img_size(1280, 720);

	//读取内外参数
	vector<Mat> matrixs;
	if (!get_calib(intrinsic_filename, extrinsic_filename, img_size, matrixs)) {
		printf("get_calib fail\n");
		return -1;
	};

	//SGBM/BM参数初始化，依据自己场景做调整
	int window_size = 5;
	int blockSize = 5;
	int minDisparity = 128;
	int numDisparities = 128 * 2;
	int P1 = 8 * 3 * window_size*window_size;
	int P2 = 32 * 3 * window_size*window_size;
	int disp12MaxDiff = 1;
	int preFilterCap = 63;
	int uniquenessRatio = 15;
	int speckleWindowSize = 0;
	int speckleRange = 2;

#if 1 //SGBM
	Ptr<StereoSGBM> left_matcher = StereoSGBM::create();
	//Ptr<StereoBM>left_matcher = StereoBM::create(16, 9);
	left_matcher->setMinDisparity(-minDisparity);
	left_matcher->setNumDisparities(numDisparities);
	left_matcher->setBlockSize(blockSize);
	left_matcher->setP1(P1);
	left_matcher->setP2(P2);
	left_matcher->setDisp12MaxDiff(disp12MaxDiff);
	left_matcher->setPreFilterCap(preFilterCap);
	left_matcher->setUniquenessRatio(uniquenessRatio);
	left_matcher->setSpeckleWindowSize(speckleWindowSize);
	left_matcher->setSpeckleRange(speckleRange);
	left_matcher->setMode(StereoSGBM::MODE_SGBM_3WAY);
#else //BM
	Ptr<StereoBM>left_matcher = StereoBM::create(16, 9);
	left_matcher->setMinDisparity(-minDisparity);
	left_matcher->setNumDisparities(numDisparities);
	left_matcher->setBlockSize(blockSize);
	left_matcher->setDisp12MaxDiff(disp12MaxDiff);
	left_matcher->setPreFilterCap(preFilterCap);
	left_matcher->setUniquenessRatio(uniquenessRatio);
	left_matcher->setSpeckleWindowSize(speckleWindowSize);
	left_matcher->setSpeckleRange(speckleRange);
#endif

	//RightMatcher 初始化
	int lmbda = 8000;
	int sigma = 1.5;
	Ptr<StereoMatcher> right_matcher = cv::ximgproc::createRightMatcher(left_matcher);
	auto wls_filter = cv::ximgproc::createDisparityWLSFilter(left_matcher);
	wls_filter->setLambda(lmbda);
	wls_filter->setSigmaColor(sigma);

	Mat img_full, left_img, right_img, left_img_remap, right_img_remap;
	Mat disp_left, disp_right, disp_end, disp_show;
	Mat threeD3;

	img_full = imread("20220804154537.jpg");
	if (img_full.empty()) {
		cout << "No img Data" << endl;
		return -1;
	}
	left_img  = img_full(Rect(0, 0, 1280, 720));
	right_img = img_full(Rect(1280, 0, 1280, 720));

	cvtColor(left_img,  left_img,  CV_BGR2GRAY);
	cvtColor(right_img, right_img, CV_BGR2GRAY);
	remap(left_img,   left_img_remap, matrixs[0], matrixs[1], INTER_LINEAR);
	remap(right_img, right_img_remap, matrixs[2], matrixs[3], INTER_LINEAR);

#if 1 //显示极线矫正后的结果
	Mat remap_full(Size(2560, 720), CV_8UC1);
	left_img_remap.copyTo(remap_full(Rect(0, 0, 1280, 720)));// = left_img_remap.clone();
	right_img_remap.copyTo(remap_full(Rect(1280, 0, 1280, 720)));// = right_img_remap.clone();

	for (int kk = 0; kk < 720; kk+=50){
		line(remap_full, Point(0, kk), Point(2560, kk), Scalar(255), 2);
	}

#endif
	left_matcher->compute(left_img_remap, right_img_remap, disp_left);
	right_matcher->compute(right_img_remap, left_img_remap, disp_right);
	wls_filter->filter(disp_left, left_img, disp_end, disp_right);

	disp_end.setTo(0, disp_end < 0);
	cv::normalize(disp_end, disp_show, 0, 255, CV_MINMAX, CV_8U);
	reprojectImageTo3D(disp_end, threeD3, matrixs[4], true); 
	threeD3 *= 16; 
	threeD3 /= 1000.; //以米为单位
	vector<Mat> split_vec;
	split(threeD3, split_vec);
	auto x = split_vec[0].clone();
	auto y = split_vec[1].clone(); 
	auto z = split_vec[2].clone();
	cv::imshow("left_img", left_img);
	cv::imshow("disp_show", disp_show);
	cv::waitKey();
}
