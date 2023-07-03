#pragma once

#include <opencv2/opencv.hpp>
#include "BoundingBox.hpp"

class Mask
{
public:
	Mask(const cv::Mat i, BoundingBox b);
	~Mask();

private:
	cv::Mat source_image;
	BoundingBox bb;

};