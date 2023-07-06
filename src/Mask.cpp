#include "Mask.hpp"
#include <map>
#include <opencv2/ximgproc/segmentation.hpp>

Mask::Mask(const cv::Mat i, const std::vector<std::pair<int, cv::Rect>> b)
	: source_image(i), bounding_boxes(b)
{
	cv::Mat txt = textureSegmentation();
	cv::imshow("texture", txt);
	cv::waitKey(0);
	return;

	// Gamma correction
	cv::Mat gc_image;
	cv::Mat lookUpTable(1, 256, CV_8U);
	uchar* p = lookUpTable.ptr();
	for (int i = 0; i < 256; ++i)
		p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, 0.5) * 255.0);
	cv::LUT(source_image, lookUpTable, gc_image);

	cv::GaussianBlur(gc_image, gc_image, cv::Size(3, 3), 0, 0);
	
	// Laplacian
	cv::Mat laplacian;
	cv::Laplacian(gc_image, laplacian, CV_8UC1, 3);
	cv::imshow("laplacian", laplacian);
	cv::waitKey(0);
	
	// threshold
	cv::Mat thresholded;
	cv::threshold(laplacian, thresholded, 86, 255, cv::THRESH_BINARY);
	cv::imshow("thresholded", thresholded);
	cv::waitKey(0);

	// only one channel from thresholded
	cv::extractChannel(thresholded, thresholded, 0);
	cv::imshow("thresholded", thresholded);
	cv::waitKey(0);

	return;

	// Saturation thresholding
	cv::Mat saturation = saturationThresholding();
	cv::imshow("saturation", saturation);
	cv::waitKey(0);

	// Texture segmentation
	cv::Mat texture = textureSegmentation();
	cv::imshow("texture", texture);
	cv::waitKey(0);

	// Combine the two masks
	cv::Mat mask = cv::Mat::zeros(source_image.size(), CV_8UC1);
	cv::bitwise_and(saturation, texture, mask);
	cv::imshow("mask", mask);
	cv::waitKey(0);
}

/**
 * @brief Saturation thresholding function, to be combined with texture segmentation
 * @return cv::Mat Mask of the saturation thresholded image
 */
cv::Mat Mask::saturationThresholding()
{
	// Gamma correction
	cv::Mat gc_image;
	cv::Mat lookUpTable(1, 256, CV_8U);
	uchar* p = lookUpTable.ptr();
	for (int i = 0; i < 256; ++i)
		p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, 0.5) * 255.0);
	cv::LUT(source_image, lookUpTable, gc_image);

	// HSV conversion
	cv::Mat hsv_image;
	cv::cvtColor(gc_image, hsv_image, cv::COLOR_BGR2HSV);

	// Saturation thresholding
	cv::Mat saturation;
	cv::extractChannel(hsv_image, saturation, 1);
	cv::threshold(saturation, saturation, 32, 255, cv::THRESH_BINARY);

	// Morphological operations
	cv::morphologyEx(saturation, saturation, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15)));
	cv::morphologyEx(saturation, saturation, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(55, 55)));

	return saturation;
}

cv::Mat Mask::textureSegmentation()
{
	// LAB conversion
	cv::Mat lab_image;
	cv::cvtColor(source_image, lab_image, cv::COLOR_BGR2Lab);

	// Channels normalization
	std::vector<cv::Mat> channels;
	cv::split(lab_image, channels);
	for (int i = 0; i < channels.size(); i++)
		cv::normalize(channels[i], channels[i], 0, 255, cv::NORM_MINMAX);

	// Channels concatenation
	cv::merge(channels, lab_image);

	// Float conversion
	lab_image.convertTo(lab_image, CV_32FC3);

	// JSEG color-texture segmentation
	cv::Ptr<cv::ximgproc::segmentation::SelectiveSearchSegmentation> jseg = cv::ximgproc::segmentation::createSelectiveSearchSegmentation();
	jseg->setBaseImage(lab_image);
	jseg->switchToSelectiveSearchFast();
	std::vector<cv::Rect> regions;
	jseg->process(regions);

	return source_image;
}

Mask::~Mask(){}

/*
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
*/

/*
cv::Mat hsv_image;
	cv::Mat lookUpTable(1, 256, CV_8U);
	uchar* p = lookUpTable.ptr();
	for (int i = 0; i < 256; ++i)
		p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, 0.5) * 255.0);
	cv::LUT(source_image, lookUpTable, hsv_image);
	cv::cvtColor(hsv_image, hsv_image, cv::COLOR_BGR2HSV);

	cv::Mat segments = cv::Mat::zeros(source_image.size(), CV_8UC1);
	for (int i = 0; i < bounding_boxes.size(); i++)
	{
		cv::Mat window;
		cv::extractChannel(hsv_image, window, 1);
		// cut the window
		for (int j = 0; j < window.rows; j++) for (int k = 0; k < window.cols; k++)
			if (j < bounding_boxes[i].second.y || j > bounding_boxes[i].second.y + bounding_boxes[i].second.height || k < bounding_boxes[i].second.x || k > bounding_boxes[i].second.x + bounding_boxes[i].second.width)
				window.at<uchar>(j, k) = 0;
		cv::threshold(window, window, 50, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
		cv::medianBlur(window, window, 5);
		cv::morphologyEx(window, window, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));
		cv::add(segments, window * 255, segments);
	}
	cv::imshow("segments", segments);
	cv::waitKey(0);
*/
