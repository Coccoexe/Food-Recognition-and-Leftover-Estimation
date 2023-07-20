#include "BoundingBoxes.hpp"

#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>

#define DEBUG false

BoundingBoxes::BoundingBoxes(const cv::Mat& input)
	: source_image(input)
{
	// Variables
	const unsigned int GAUSSIAN_BLUR_KERNEL_SIZE = 5;
	const unsigned int HOUGH_CANNY_THRESHOLD = 60;
	const unsigned int HOUGH_CIRCLE_ROUNDNESS = 70;
	const unsigned int PLATES_MIN_RADIUS = 240;
	const unsigned int PLATES_MAX_RADIUS = 325;
	const unsigned int BOWL_MIN_RADIUS = 170;
	const unsigned int BOWL_MAX_RADIUS = 220;
	const unsigned int MIN_DISTANCE_BETWEEN_CIRCLES = 300;
	const unsigned int BREAD_FACTOR = 3;
	cv::Mat debug_image;
	if (DEBUG) debug_image = source_image.clone();

	// Grayscale image
	cv::Mat grayscale_image;
	cv::cvtColor(source_image, grayscale_image, cv::COLOR_BGR2GRAY);
	cv::GaussianBlur(grayscale_image, grayscale_image, cv::Size(GAUSSIAN_BLUR_KERNEL_SIZE, GAUSSIAN_BLUR_KERNEL_SIZE), 2, 2);

	// 1. Detect plates
	std::vector<cv::Vec3f> plates_circles;
	cv::HoughCircles(grayscale_image, plates_circles, cv::HOUGH_GRADIENT, 1, MIN_DISTANCE_BETWEEN_CIRCLES, HOUGH_CANNY_THRESHOLD, HOUGH_CIRCLE_ROUNDNESS, PLATES_MIN_RADIUS, PLATES_MAX_RADIUS);
	if (DEBUG) for (const auto& circle : plates_circles) cv::circle(debug_image, cv::Point(cvRound(circle[0]), cvRound(circle[1])), cvRound(circle[2]), cv::Scalar(255, 0, 0), 2);

	// 2. Detect salad (if exists)
	std::vector<cv::Vec3f> salad_circles;
	cv::HoughCircles(grayscale_image, salad_circles, cv::HOUGH_GRADIENT, 1, MIN_DISTANCE_BETWEEN_CIRCLES, HOUGH_CANNY_THRESHOLD, HOUGH_CIRCLE_ROUNDNESS, BOWL_MIN_RADIUS, BOWL_MAX_RADIUS);
	if (DEBUG) for (const auto& circle : salad_circles) cv::circle(debug_image, cv::Point(cvRound(circle[0]), cvRound(circle[1])), cvRound(circle[2]), cv::Scalar(0, 255, 0), 2);

	// Save results
	plates = plates_circles;
	!salad_circles.empty() ? salad = std::make_pair(true, salad_circles[0]) : salad = std::make_pair(false, cv::Vec3f());

	// 3. Detect bread (if exists)
	auto find_bread = [](const cv::Mat& source_image, const std::vector<cv::Vec3f>& plates, std::pair<bool, cv::Vec3f>& salad) -> std::pair<bool, cv::Mat>
	{
		// Variables
		const unsigned int CLOSE_KERNEL_SIZE = 9;
		const unsigned int DILATE_KERNEL_SIZE = 5;
		const unsigned int MIN_AREA_THRESHOLD = 6000;
		const unsigned int MAX_AREA_THRESHOLD = 60000;
		const unsigned int CIRCLE_NEIGHBORHOOD = 5;
		auto gamma_correction = [](const cv::Mat& input) -> cv::Mat
		{
			// Gamma correction
			cv::Mat gc_image;
			cv::Mat lookUpTable(1, 256, CV_8U);
			uchar* p = lookUpTable.ptr();
			for (int i = 0; i < 256; ++i)
				p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, 0.5) * 255.0);
			cv::LUT(input, lookUpTable, gc_image);

			return gc_image;
		};
		auto saturation_thresholding = [](const cv::Mat& input) -> cv::Mat
		{
			// Variables
			const unsigned int SATURATION_THRESHOLD = 30;

			// Convert to HSV
			cv::Mat hsv_image;
			cv::cvtColor(input, hsv_image, cv::COLOR_BGR2HSV);
			cv::Mat saturation;
			cv::extractChannel(hsv_image, saturation, 1);

			// Thresholding
			cv::threshold(saturation, saturation, SATURATION_THRESHOLD, 255, cv::THRESH_BINARY);

			return saturation;
		};
		auto niBlack_thresholding = [](const cv::Mat& input) -> cv::Mat
		{
			// Variables
			const unsigned int NIBLACK_BLOCK_SIZE = 19;
			const double NIBLACK_K = 0.7;

			// Convert to grayscale
			cv::Mat gray_image;
			cv::cvtColor(input, gray_image, cv::COLOR_BGR2GRAY);

			// Thresholding
			cv::Mat niblack;
			cv::ximgproc::niBlackThreshold(gray_image, niblack, 255, cv::THRESH_BINARY, NIBLACK_BLOCK_SIZE, NIBLACK_K, cv::ximgproc::BINARIZATION_NIBLACK);

			return niblack;
		};
		auto fill_holes = [](cv::Mat& input) -> void
		{
			cv::Mat result = input.clone();

			// Fill holes
			cv::floodFill(result, cv::Point(0, 0), 255);
			cv::Mat inversed;
			cv::bitwise_not(result, inversed);
			input = (input | inversed);
		};
		auto filter_areas = [](const cv::Mat& input, cv::Mat& output, const unsigned int threshold) -> void
		{
			std::vector<std::vector<cv::Point>> c;

			cv::findContours(input.clone(), c, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
			for (int i = 0; i < c.size(); i++)
				if (cv::contourArea(c[i]) > threshold)
					cv::drawContours(output, c, i, 255, -1);
		};
		auto remove_outliers = [](const cv::Mat& input, cv::Mat& output, cv::Rect& output_box) -> bool
		{
			std::vector<std::vector<cv::Point>> contours;
			cv::findContours(input.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

			double max_area = 0;
			int max_area_index = -1;
			for (int i = 0; i < contours.size(); i++)
			{
				// Remove contours that are too dense
				cv::Rect box = cv::boundingRect(contours[i]);
				double area = cv::contourArea(contours[i]);
				if (area / box.area() > 0.7)
					continue;

				// Remove contours that touch the border
				bool touches = false;
				for (const auto& point : contours[i])
					if (point.x == 0 || point.x == input.cols - 1 || point.y == 0 || point.y == input.rows - 1)
					{
						touches = true;
						break;
					}
				if (touches)
					continue;

				// Remove contours that are too elongated
				if (box.width / box.height > 3 || box.height / box.width > 3)
					continue;

				// Remove contours that have one dimension too big / small
				if (box.width > input.cols / 4 || box.height > input.rows / 2.75 || box.width < 100 || box.height < 100)
					continue;

				// Save contour
				if (area > max_area)
				{
					max_area = area;
					max_area_index = i;
				}
			}
			if (max_area_index == -1)
				return false;
			cv::drawContours(output, contours, max_area_index, 255, -1);
			output_box = cv::boundingRect(contours[max_area_index]);
			return true;
		};

		// Remove plates and salad from image
		cv::Mat image = source_image.clone();
		for (const auto& plate : plates)
			cv::circle(image, cv::Point(cvRound(plate[0]), cvRound(plate[1])), cvRound(plate[2]), cv::Scalar(0, 0, 0), -1);
		if (salad.first)
			cv::circle(image, cv::Point(cvRound(salad.second[0]), cvRound(salad.second[1])), cvRound(salad.second[2]), cv::Scalar(0, 0, 0), -1);

		// Saturation thresholding & niBlack thresholding
		cv::Mat gc_image = gamma_correction(image);
		cv::Mat saturation = saturation_thresholding(gc_image);
		cv::Mat niblack = niBlack_thresholding(gc_image);
		cv::Mat mask = saturation & niblack;

		// Morphological operations
		cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE)));
		cv::dilate(mask, mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(DILATE_KERNEL_SIZE, DILATE_KERNEL_SIZE)));
		fill_holes(mask);

		// Filter areas
		cv::Mat nosmall = cv::Mat::zeros(mask.size(), CV_8UC1), yesbig = cv::Mat::zeros(mask.size(), CV_8UC1);
		filter_areas(mask, nosmall, MIN_AREA_THRESHOLD);
		filter_areas(mask, yesbig, MAX_AREA_THRESHOLD);
		cv::Mat filtered = nosmall - yesbig;

		// Remove outliers
		cv::Mat no_outliers = cv::Mat::zeros(filtered.size(), CV_8UC1);
		cv::Rect box;
		if (!remove_outliers(filtered, no_outliers, box))
			return std::make_pair(false, cv::Mat());

		// Perform grabCut segmentation
		cv::Mat bgd_mask = cv::Mat::zeros(image.size(), CV_8UC1);
		bgd_mask(box) = cv::GC_PR_BGD;
		cv::threshold(no_outliers, no_outliers, 0, 1, cv::THRESH_BINARY);
		cv::Mat grabcut_mask = cv::Mat::zeros(image.size(), CV_8UC1);
		cv::addWeighted(no_outliers, 1, bgd_mask, 1, 0, grabcut_mask);
		cv::Mat bgd_model, fgd_model;
		cv::grabCut(image, grabcut_mask, box, bgd_model, fgd_model, 5, cv::GC_INIT_WITH_MASK);
		cv::Mat1b result_mask = (grabcut_mask == cv::GC_PR_FGD) | (grabcut_mask == cv::GC_FGD);

		// Check if 'result_mask' white area is too close or touches plates
		cv::Mat diff = cv::Mat::zeros(result_mask.size(), CV_8UC1);
		for (auto& plate : plates)
			cv::circle(diff, cv::Point(cvRound(plate[0]), cvRound(plate[1])), cvRound(plate[2]) + CIRCLE_NEIGHBORHOOD, cv::Scalar(255, 255, 255), -1);
		if (salad.first)
			cv::circle(diff, cv::Point(cvRound(salad.second[0]), cvRound(salad.second[1])), cvRound(salad.second[2]) + CIRCLE_NEIGHBORHOOD, cv::Scalar(255, 255, 255), -1);
		if (cv::countNonZero(diff & result_mask) > 0)
			return std::make_pair(false, cv::Mat());

		return std::make_pair(true, result_mask);
	};
	bread = find_bread(source_image, plates_circles, salad);
	if (DEBUG && bread.first) debug_image.setTo(cv::Scalar(200, 200, 0), bread.second);

	// Show debug image
	if (DEBUG) { cv::imshow("DEBUG: Bounding Boxes", debug_image); cv::waitKey(0); };
}