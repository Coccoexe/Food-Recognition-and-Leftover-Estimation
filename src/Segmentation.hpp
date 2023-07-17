#pragma once

#include <vector>

#include <opencv2/opencv.hpp>

class Segmentation
{
public:
	Segmentation(cv::Mat& p, std::vector<int> l);
	cv::Mat getSegments() const { return segments; }

private:
	cv::Mat plate;
	const std::vector<int> labels;
	cv::Mat segments;
	void correction(cv::Mat& in, cv::Mat& out);
	void process(cv::Mat& ranged, cv::Mat& out);

	//rgb min and max ranges
	const std::vector<std::pair<cv::Scalar,cv::Scalar>> c_ranges = {
		std::make_pair<cv::Scalar,cv::Scalar>(cv::Scalar(0,0,0), cv::Scalar(0,0,0)),			// black
		std::make_pair<cv::Scalar,cv::Scalar>(cv::Scalar(0,105,150), cv::Scalar(16,168,255)),	// pasta with pesto
		std::make_pair<cv::Scalar,cv::Scalar>(cv::Scalar(0,139,153), cv::Scalar(52,255,255)),	// pasta with tomato sauce
		std::make_pair<cv::Scalar,cv::Scalar>(cv::Scalar(0,127,100), cv::Scalar(24,246,238)),	// pasta with meat sauce
		std::make_pair<cv::Scalar,cv::Scalar>(cv::Scalar(0,54,123), cv::Scalar(9,180,255)),		// pasta with clams and mussels
		std::make_pair<cv::Scalar,cv::Scalar>(cv::Scalar(56,153,187), cv::Scalar(101,191,245)), // pilaw rice with peppers and peas
		std::make_pair<cv::Scalar,cv::Scalar>(cv::Scalar(0,125,191), cv::Scalar(158,215,255)),	// grilled pork cutlet
		std::make_pair<cv::Scalar,cv::Scalar>(cv::Scalar(0,0,205), cv::Scalar(101,147,255)),	// fish cutlet
		std::make_pair<cv::Scalar,cv::Scalar>(cv::Scalar(0,0,0), cv::Scalar(0,0,0)),			// rabbit
		std::make_pair<cv::Scalar,cv::Scalar>(cv::Scalar(0,31,177), cv::Scalar(87,118,255)),	// seafood salad
		std::make_pair<cv::Scalar,cv::Scalar>(cv::Scalar(0,0,137), cv::Scalar(203,57,252)),			// beans
		std::make_pair<cv::Scalar,cv::Scalar>(cv::Scalar(0,156,158), cv::Scalar(104,255,219)),	// basil potatoes
		std::make_pair<cv::Scalar,cv::Scalar>(cv::Scalar(0,0,0), cv::Scalar(0,0,0)),			// salad
		std::make_pair<cv::Scalar,cv::Scalar>(cv::Scalar(0,0,0), cv::Scalar(0,0,0))				// bread
	};
};

