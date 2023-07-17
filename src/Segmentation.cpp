# include "Segmentation.hpp"

Segmentation::Segmentation(cv::Mat& p, std::vector<std::string> l)
	: plate(p), labels(l)
{
	segments = cv::Mat::zeros(plate.size(), CV_8UC1);
}

cv::Mat Segmentation::process(cv::Mat ranged, cv::Mat colored)
{
	auto filterAreas = [](const cv::Mat& input, cv::Mat& output, const unsigned int threshold) -> void
	{
		std::vector<std::vector<cv::Point>> c;

		cv::findContours(input.clone(), c, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
		for (int i = 0; i < c.size(); i++)
			if (cv::contourArea(c[i]) > threshold)
				cv::drawContours(output, c, i, 255, -1);
	};

	auto fillHoles = [](cv::Mat& input) -> void
	{
		cv::Mat ff = input.clone();
		cv::floodFill(ff, cv::Point(0, 0), cv::Scalar(255));
		cv::Mat inversed_ff;
		cv::bitwise_not(ff, inversed_ff);
		input = (input | inversed_ff);
	};
	//median
	cv::medianBlur(ranged, ranged, 5);

	//closing
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(40, 40));
	cv::morphologyEx(ranged, ranged, cv::MORPH_CLOSE, kernel);

	//dilation
	cv::Mat a = cv::Mat::zeros(ranged.size(), CV_8UC1);
	filterAreas(ranged, a, 8000);
	cv::dilate(a, a, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15)));

	//closing
	kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15));
	cv::morphologyEx(a, a, cv::MORPH_CLOSE, kernel);

	//filling holes
	fillHoles(a);

	//detected
	cv::Mat original;
	cv::bitwise_and(colored, colored, original, a);

	return a;
}