#pragma once

#include "BoundingBox.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

class Mask
{
public:
	Mask(const cv::Mat i);
	~Mask();

private:
	const cv::Mat source_image;

	cv::Mat saturationThresholding();
	cv::Mat textureSegmentation();
};