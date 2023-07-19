#pragma once

#include <vector>

#include <opencv2/opencv.hpp>

class Metrics
{
public:
	Metrics(cv::Mat& m, std::vector<std::pair<int, cv::Rect>> b, cv::Mat& o_m, std::vector<std::pair<int, cv::Rect>> o_b);

private:
	cv::Mat mask, orig_mask;
	const std::vector<std::pair<int, cv::Rect>> boxes, orig_boxes;
};