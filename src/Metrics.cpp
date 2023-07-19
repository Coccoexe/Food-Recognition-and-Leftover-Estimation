#include "Metrics.hpp"

Metrics::Metrics(cv::Mat& m, std::vector<std::pair<int, cv::Rect>> l, cv::Mat& o_m, std::vector<std::pair<int, cv::Rect>> o_b)
	: mask(m), boxes(l), orig_mask(o_m), orig_boxes(o_b)
{
	
	for (int i = 0; i < boxes.size(); i++)
	{
		std::cout << "box      " << i << ": " << boxes[i].second << std::endl;
		std::cout << "orig_box " << i << ": " << orig_boxes[i].second << std::endl;
	}

	cv::imshow("mask", mask);
	cv::imshow("orig_mask", orig_mask);
	cv::waitKey(0);
}
