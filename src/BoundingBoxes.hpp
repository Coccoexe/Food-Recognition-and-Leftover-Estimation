#pragma once

#include <vector>

#include <opencv2/opencv.hpp>

class BoundingBoxes
{
public:
	BoundingBoxes(const cv::Mat& input);
	std::vector<cv::Vec3f> getPlates() const { return plates; }
	std::pair<bool, cv::Vec3f> getSalad() const { return salad; }
	std::pair<bool, cv::Rect> getBread() const { return bread; }

private:
	const cv::Mat source_image;
	std::vector<cv::Vec3f> plates;    // [bounding boxes]
	std::pair<bool, cv::Vec3f> salad; // <found, bounding box>
	std::pair<bool, cv::Rect> bread;  // <found, bounding box>
};