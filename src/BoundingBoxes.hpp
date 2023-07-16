#pragma once

#include <vector>

#include <opencv2/opencv.hpp>

class BoundingBoxes
{
public:
	BoundingBoxes(const cv::Mat& input);
	std::vector<cv::Rect> getPlates() const { return plates; }
	std::pair<bool, cv::Rect> getSalad() const { return salad; }
	std::pair<bool, cv::Rect> getBread() const { return bread; }

private:
	const cv::Mat source_image;
	std::vector<cv::Rect> plates;    // [bounding boxes]
	std::pair<bool, cv::Rect> salad; // <found, bounding box>
	std::pair<bool, cv::Rect> bread; // <found, bounding box>
};