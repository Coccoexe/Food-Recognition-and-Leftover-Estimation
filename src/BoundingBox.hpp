#pragma once

#include <opencv2/opencv.hpp>
#include <map>
#include <vector>

class BoundingBox
{
public:
	BoundingBox(const cv::Mat i, const std::vector<std::string> l, std::multimap<int, const cv::Mat> m);
	~BoundingBox();

private:
	const cv::Mat source_image;
	const std::vector<std::string> labels;
	const std::multimap<int, const cv::Mat> match;
};