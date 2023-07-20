#pragma once

#include <vector>

#include <opencv2/opencv.hpp>

class BoundingBoxes
{
public:
	/**
	 * @brief Construct a new Bounding Boxes object, detecting the general location of the plates, the salad and the bread.
	 * @param input The input image.
	 */
	BoundingBoxes(const cv::Mat& input);
	std::vector<cv::Vec3f> getPlates() const { return plates; }
	std::pair<bool, cv::Vec3f> getSalad() const { return salad; }
	std::pair<bool, cv::Mat> getBread() const { return bread; }

private:
	const cv::Mat source_image;
	std::vector<cv::Vec3f> plates;      // [circles]
	std::pair<bool, cv::Vec3f> salad;   // <found, circle>
	std::pair<bool, cv::Mat> bread;     // <found, bounding box>
};