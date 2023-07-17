# include "Segmentation.hpp"

Segmentation::Segmentation(cv::Mat& p, std::vector<std::string> l)
	: plate(p), labels(l)
{
	segments = cv::Mat::zeros(plate.size(), CV_8UC1);
}
