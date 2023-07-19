#include "Metrics.hpp"

#define DEBUG false

Metrics::Metrics(cv::Mat& m, std::vector<std::pair<int, cv::Rect>> l, cv::Mat& o_m, std::vector<std::pair<int, cv::Rect>> o_b)
	: mask(m), boxes(l), orig_mask(o_m), orig_boxes(o_b)
{
	if (DEBUG)
	{
		for (int i = 0; i < boxes.size(); i++)
		{
			std::cout << "box      " << boxes[i].first << ": " << boxes[i].second << std::endl;
			std::cout << "orig_box " << boxes[i].first << ": " << orig_boxes[i].second << std::endl;
		}

		cv::imshow("mask", mask);
		cv::imshow("orig_mask", orig_mask);
		cv::waitKey(0);
	}
}
