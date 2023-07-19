#pragma once

#include <vector>

#include <opencv2/opencv.hpp>

class Segmentation
{
public:
	Segmentation(cv::Mat& p, std::vector<int> l);
	cv::Mat getSegments() const { return segments; }
	std::vector<std::pair<int, int>> getAreas() const { return areas; }
	std::vector<std::pair<int, cv::Rect>> getBoxes() const { return boxes; }

private:
	cv::Mat plate;
	const std::vector<int> labels;
	cv::Mat segments;
	std::vector<std::pair<int, cv::Rect>> boxes;
	std::vector<std::pair<int, int>> areas;
	void correction(cv::Mat& in, cv::Mat& out);
	void process(cv::Mat& ranged, cv::Mat& out);

	//rgb min and max ranges
	const std::vector<std::pair<cv::Scalar,cv::Scalar>> c_ranges = {
		std::make_pair<cv::Scalar,cv::Scalar>(cv::Scalar(0,0,0), cv::Scalar(0,0,0)),			// black
		std::make_pair<cv::Scalar,cv::Scalar>(cv::Scalar(2,109,168), cv::Scalar(59,167,255)),	// pasta with pesto
		std::make_pair<cv::Scalar,cv::Scalar>(cv::Scalar(0,139,153), cv::Scalar(52,255,255)),	// pasta with tomato sauce
		std::make_pair<cv::Scalar,cv::Scalar>(cv::Scalar(12,110,153), cv::Scalar(30,172,212)),	// pasta with meat sauce
		std::make_pair<cv::Scalar,cv::Scalar>(cv::Scalar(0,54,123), cv::Scalar(9,180,255)),		// pasta with clams and mussels
		std::make_pair<cv::Scalar,cv::Scalar>(cv::Scalar(56,153,187), cv::Scalar(101,191,245)), // pilaw rice with peppers and peas
		std::make_pair<cv::Scalar,cv::Scalar>(cv::Scalar(45,128,187),cv::Scalar(108,161,215)),	// grilled pork cutlet
		std::make_pair<cv::Scalar,cv::Scalar>(cv::Scalar(0,0,205), cv::Scalar(101,147,255)),	// fish cutlet
		std::make_pair<cv::Scalar,cv::Scalar>(cv::Scalar(0,68,154), cv::Scalar(38,108,234)),	// rabbit
		std::make_pair<cv::Scalar,cv::Scalar>(cv::Scalar(0,28,139), cv::Scalar(62,92,255)),		// seafood salad
		std::make_pair<cv::Scalar,cv::Scalar>(cv::Scalar(14,38,0),cv::Scalar(24,71,180)),		// beans
		std::make_pair<cv::Scalar,cv::Scalar>(cv::Scalar(0,156,158), cv::Scalar(104,255,219)),	// basil potatoes
		std::make_pair<cv::Scalar,cv::Scalar>(cv::Scalar(0,0,0), cv::Scalar(0,0,0)),			// salad //for CLIP this is empty plate
		std::make_pair<cv::Scalar,cv::Scalar>(cv::Scalar(0,0,0), cv::Scalar(0,0,0))				// bread //never appears in CLIP
	};
};
