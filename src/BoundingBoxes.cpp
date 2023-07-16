#include "BoundingBoxes.hpp"

#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#define DEBUG true

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

	// 3. Detect bread (if exists)
	// TODO: implement

	// Show debug image
	if (DEBUG) { cv::imshow("DEBUG: Bounding Boxes", debug_image); cv::waitKey(0); }

	// Save results
	plates = plates_circles;
	!salad_circles.empty() ? salad = std::make_pair(true, salad_circles[0]) : salad = std::make_pair(false, cv::Vec3f());
}