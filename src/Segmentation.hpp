#pragma once

#include <vector>

#include <opencv2/opencv.hpp>

class Segmentation
{
public:
	Segmentation(cv::Mat& p, std::vector<std::string> l);
	cv::Mat getSegments() const { return segments; }

private:
	const cv::Mat plate;
	const std::vector<std::string> labels;
	cv::Mat segments;

};

