#pragma once

#include <vector>

#include <opencv2/opencv.hpp>

class Metrics
{
public:
	Metrics(std::vector<std::vector<std::tuple<cv::Mat,std::vector<std::pair<int, cv::Rect>>,cv::Mat,std::vector<std::pair<int, cv::Rect>>>>>& m);

private:
	const std::vector<std::vector<std::tuple<cv::Mat, std::vector<std::pair<int, cv::Rect>>, cv::Mat, std::vector<std::pair<int, cv::Rect>>>>> metrics;
	std::vector<int> false_positives;
	std::vector<int> false_negatives;
	std::vector<int> true_positives;
	std::vector<std::vector<double>> precision;
	std::vector<std::vector<double>> recall;
	std::vector<double> average_precision;
};