# include "Segmentation.hpp"

Segmentation::Segmentation(cv::Mat& p, std::vector<int> l)
	: plate(p), labels(l)
{
	segments = cv::Mat::zeros(plate.size(), CV_8UC1);

	//correction
	cv::Mat corrected;
	correction(plate, corrected);
	cv::imshow("corrected", corrected);

	//segmentation
	for (const auto label : labels)
	{
		cv::Mat ranged, mask;
		cv::inRange(corrected, c_ranges[label].first, c_ranges[label].second, ranged);
		//cv::imshow("ranged", ranged);
		process(ranged, mask);
		cv::threshold(mask, mask, 0, label*15, cv::THRESH_BINARY);
		segments = segments | mask;
	}
	cv::imshow("segments", segments);
	cv::waitKey(0);
}

void Segmentation::correction(cv::Mat& in, cv::Mat& out)
{
	//gamma transform
	cv::Mat gamma;
	cv::Mat lookUpTable(1, 256, CV_8U);
	uchar* p = lookUpTable.ptr();
	double gamma_ = 0.5;
	for (int i = 0; i < 256; ++i)
		p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, gamma_) * 255.0);
	cv::LUT(in, lookUpTable, gamma);

	//image to hsv
	cv::Mat hsv;
	cv::cvtColor(gamma, hsv, cv::COLOR_BGR2HSV);
	std::vector<cv::Mat> hsv_channels;
	cv::split(hsv, hsv_channels);
	cv::equalizeHist(hsv_channels[1], hsv_channels[1]);
	cv::merge(hsv_channels, out);
	cv::cvtColor(out, out, cv::COLOR_HSV2BGR);
	return;
}

void Segmentation::process(cv::Mat& in, cv::Mat& out)
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
	cv::medianBlur(in, in, 5);

	//closing
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(30, 30)); //changed from 40x40
	cv::morphologyEx(in, in, cv::MORPH_CLOSE, kernel);

	//dilation
	out = cv::Mat::zeros(in.size(), CV_8UC1);
	filterAreas(in, out, 8000);
	cv::dilate(out, out, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(20, 20))); //changed from 15x15

	//closing
	kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(25, 25)); //changed from 15x15
	cv::morphologyEx(out, out, cv::MORPH_CLOSE, kernel);

	//filling holes
	fillHoles(out);

	//cv::Mat original;
	//cv::bitwise_and(plate, plate, original, out);
	//cv::imshow("original", original);
	//cv::waitKey(0);

	return;
}