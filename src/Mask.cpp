#include "Mask.hpp"

Mask::Mask(const cv::Mat i, const std::vector<std::pair<int, cv::Rect>> b)
	: source_image(i), bounding_boxes(b)
{
	cv::Mat segments = cv::Mat::zeros(source_image.size(), CV_8UC1);
	for (int i = 0; i < bounding_boxes.size(); i++)
	{   // GrabCut for each bounding box
		cv::Mat window = cv::Mat::zeros(source_image.size(), CV_8UC1);
		cv::Mat bgdModel, fgdModel;
		cv::grabCut(source_image, window, bounding_boxes[i].second, bgdModel, fgdModel, 3, cv::GC_INIT_WITH_RECT);
		cv::compare(window, cv::GC_PR_FGD, window, cv::CMP_EQ);
		//cv::medianBlur(window, window, 3);
		cv::erode(window, window, cv::Mat(), cv::Point(-1, -1), 3);
		cv::dilate(window, window, cv::Mat(), cv::Point(-1, -1), 3);
		cv::dilate(window, window, cv::Mat(), cv::Point(-1, -1), 3);
		cv::erode(window, window, cv::Mat(), cv::Point(-1, -1), 3);
		cv::add(segments, window * bounding_boxes[i].first / 13, segments);
	}
	cv::imshow("segments", segments);
	cv::waitKey(0);
}

Mask::~Mask(){}