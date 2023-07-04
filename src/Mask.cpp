#include "Mask.hpp"

Mask::Mask(const cv::Mat i, const std::vector<std::pair<int, cv::Rect>> b)
	: source_image(i), bounding_boxes(b)
{
	cv::Mat segments = cv::Mat::zeros(source_image.size(), CV_8UC1);
	for (int i = 0; i < bounding_boxes.size(); i++)
	{   // GrabCut for each bounding box
		cv::Mat window = cv::Mat::zeros(source_image.size(), CV_8UC1);                                                // Mask for the current bounding box
		cv::Mat bgdModel, fgdModel;                                                                                   // Background and foreground models
		cv::grabCut(source_image, window, bounding_boxes[i].second, bgdModel, fgdModel, 3, cv::GC_INIT_WITH_RECT);    // GrabCut
		cv::compare(window, cv::GC_PR_FGD, window, cv::CMP_EQ);                                                       // Keep only the foreground
		cv::medianBlur(window, window, 5);                                                                            // Median blur to remove noise and fill holes
		cv::morphologyEx(window, window, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5))); // Morphological closing to fill holes
		cv::add(segments, window * bounding_boxes[i].first / 13, segments);                                           // Add the current mask to the final mask
	}
	cv::imshow("segments", segments);
	cv::waitKey(0);
}

Mask::~Mask(){}

