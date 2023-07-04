#pragma once

#include "BoundingBox.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

class Mask
{
public:
	Mask(const cv::Mat i, const std::vector<std::pair<int, cv::Rect>> b);
	~Mask();

private:
	const cv::Mat source_image;
	const std::vector<std::pair<int, cv::Rect>> bounding_boxes;
};