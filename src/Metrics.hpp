#pragma once

#include <vector>

#include <opencv2/opencv.hpp>

class Metrics
{
public:
    /**
     * @brief Construct a new Metrics object and calculate the metrics, as requested in the assignment.
     * @param m The vector of metrics, as computed in src\main.cpp.
     */
    Metrics(std::vector<std::vector<std::tuple<cv::Mat, std::vector<std::pair<int, cv::Rect>>, cv::Mat, std::vector<std::pair<int, cv::Rect>>>>>& m);

private:
    const std::vector<std::vector<std::tuple<cv::Mat, std::vector<std::pair<int, cv::Rect>>, cv::Mat, std::vector<std::pair<int, cv::Rect>>>>> metrics;
    std::vector<double> false_positives;
    std::vector<double> false_negatives;
    std::vector<double> true_positives;
    std::vector<std::vector<double>> precision;
    std::vector<std::vector<double>> recall;
    std::vector<std::vector<double>> IoU;
    std::vector<double> average_precision;
};