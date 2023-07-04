#include "BoundingBox.hpp"

BoundingBox::BoundingBox(const cv::Mat i, const std::vector<std::string> l, std::multimap<int, const cv::Mat> m)
	: source_image(i), labels(l), match(m)
{	// Compute bounding boxes and labeling
	//cv::Mat grayscale_image = cv::cvtColor(source_image, cv::COLOR_BGR2GRAY);
}

BoundingBox::~BoundingBox()
{
}