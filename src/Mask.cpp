#include "Mask.hpp"
#include <map>

Mask::Mask(const cv::Mat i, const std::vector<std::pair<int, cv::Rect>> b)
	: source_image(i), bounding_boxes(b)
{
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
	// Grayscale conversion
	cv::Mat gray_image;
	cv::cvtColor(source_image, gray_image, cv::COLOR_BGR2GRAY);

	// Entropy calculation
	double pixel_entropy_value;
	cv::Mat entropy;
	const int HIST_DIM = 9;
	const int HIST_OFFSET = (HIST_DIM - 1) / 2;
	const cv::Point OFFSET_POINT = cv::Point(HIST_OFFSET, HIST_OFFSET);
	bool stop = false;
	cv::MatIterator_<float> it = entropy.begin<float>();
	cv::Mat hist;

	// Initial histogram
	int hist_size = 256;
	float range[] = { 0, 256 };
	const float* hist_range = { range };
	cv::Mat initial_hist_mask = cv::Mat::ones(HIST_DIM, HIST_DIM, CV_8UC1);
	cv::copyMakeBorder(initial_hist_mask, initial_hist_mask, 0, gray_image.rows - HIST_DIM, 0, gray_image.cols - HIST_DIM, cv::BORDER_CONSTANT, 0);
	cv::calcHist(&gray_image, 1, 0, initial_hist_mask, hist, 1, &hist_size, &hist_range, true, false);

	// Entropy calculation
	int direction = 1;
	cv::Point current_point(HIST_OFFSET, HIST_OFFSET);
	pixel_entropy_value = 0;
	cv::MatConstIterator_<float> hist_it;
	for (hist_it = hist.begin<float>(); hist_it != hist.end<float>(); hist_it++)
		if (*hist_it > 0)
			pixel_entropy_value -= *hist_it * log2(*hist_it);
	pixel_entropy_value /= 255;
	*it = pixel_entropy_value;
	cv::Point previous_point = current_point;
	current_point.y++;
	std::map<unsigned int, int> m;

	// Loop
	while (!stop)
	{
		// Get current pointers
		it = entropy.begin<float>() + (current_point - OFFSET_POINT).x * entropy.cols + (current_point - OFFSET_POINT).y;

		// Calculate the histogram (update)
		if (previous_point.y != current_point.y)
		{
			return entropy;
		}
	}
	return entropy;
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
