#include "Mask.hpp"
#include <map>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc/segmentation.hpp>

Mask::Mask(const cv::Mat i)
	: source_image(i)
{
	// Declarations
	const unsigned int AREA_THRESHOLD = 2000;
	const unsigned int CONTOURS_DISTANCE_THRESHOLD = 50;
	const unsigned int CANNY_THRESHOLD = 175;
	const unsigned int CIRCLE_ROUNDNESS = 70;
	typedef std::vector<cv::Point> Contour;
	typedef std::vector<Contour> Contours;
	auto filterAreas = [](const cv::Mat& input, cv::Mat& output, const unsigned int threshold) -> void
	{
		Contours c;

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

	// Saturation thresholding
	cv::Mat saturation = saturationThresholding();
	cv::imshow("saturation", saturation);
	cv::waitKey(0);

	// Texture segmentation
	cv::Mat texture = textureSegmentation();
	cv::imshow("texture", texture);
	cv::waitKey(0);

	// Combine the two masks
	cv::Mat m = cv::Mat::zeros(source_image.size(), CV_8UC1);
	cv::bitwise_and(saturation, texture, m);
	cv::Mat mask = cv::Mat::zeros(m.size(), CV_8UC1);
	filterAreas(m, mask, AREA_THRESHOLD);
	cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9)));
	fillHoles(mask);
	cv::imshow("mask", mask);
	cv::waitKey(0);

	// Keep stuff outside plates
	cv::Mat gs_image;
	cv::cvtColor(source_image, gs_image, cv::COLOR_BGR2GRAY);
	cv::GaussianBlur(gs_image, gs_image, cv::Size(9, 9), 2, 2);
	std::vector<cv::Vec3f> circles;
	cv::HoughCircles(gs_image, circles, cv::HOUGH_GRADIENT, 1, gs_image.rows / 16, CANNY_THRESHOLD, CIRCLE_ROUNDNESS, 0, 0);

	cv::Mat dishes = source_image.clone();
	for (int i = 0; i < circles.size(); i++)
		cv::circle(dishes, cv::Point(circles[i][0], circles[i][1]), circles[i][2], cv::Scalar(0, 0, 255), 3);
	cv::imshow("dishes", dishes);
	cv::waitKey(0);


	cv::Mat side_mask = mask.clone();
	for (int i = 0; i < circles.size(); i++)
		cv::circle(side_mask, cv::Point(circles[i][0], circles[i][1]), circles[i][2], 0, -1);
	
	// Find contours outside plates
	Contours contours;
	cv::findContours(side_mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	// Link close contours based on CONTOURS_DISTANCE_THRESHOLD
	for (int i = 0; i < contours.size(); i++)
		for (int j = i + 1; j < contours.size(); j++)
			for (int k = 0; k < contours[i].size(); k++)
				for (int l = 0; l < contours[j].size(); l++)
					if (abs(contours[i][k].x - contours[j][l].x) < CONTOURS_DISTANCE_THRESHOLD && abs(contours[i][k].y - contours[j][l].y) < CONTOURS_DISTANCE_THRESHOLD)
						cv::line(side_mask, contours[i][k], contours[j][l], 255, 1);
						
	// Find contours again
	contours.clear();
	cv::findContours(side_mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	if (contours.size() > 1)
	{	// We have more than one contour, find the largest one
		int largest_contour_index = 0;
		for (int i = 0; i < contours.size(); i++)
			if (cv::contourArea(contours[i]) >= cv::contourArea(contours[largest_contour_index]))
				largest_contour_index = i;
		// Keep only the largest contour
		cv::Mat single_side_mask = cv::Mat::zeros(side_mask.size(), CV_8UC1);
		cv::drawContours(single_side_mask, contours, largest_contour_index, 255, -1);
		// Back to the mask
		//cv::imshow("single_side_mask", single_side_mask);
		//cv::imshow("side_mask", side_mask);
		//cv::imshow("mask", mask);
		//cv::waitKey(0);
		mask = mask - (side_mask - single_side_mask);
	}
	else
	{	// We have only one contour, keep it
		//cv::imshow("side_mask", side_mask);
		//cv::imshow("mask", mask);
		//cv::waitKey(0);
		mask = mask + side_mask;
	}
	
	// Find all contours
	cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	// Find bounding boxes
	std::vector<cv::Rect> bounding_boxes;
	for (int i = 0; i < contours.size(); i++)
		bounding_boxes.push_back(cv::boundingRect(contours[i]));

	// Draw bounding boxes
	cv::Mat bounding_boxes_image = source_image.clone();
	for (int i = 0; i < bounding_boxes.size(); i++)
		cv::rectangle(bounding_boxes_image, bounding_boxes[i], cv::Scalar(0, 255, 0), 2);
	cv::imshow("bounding_boxes", bounding_boxes_image);
	cv::waitKey(0);

}

/**
 * @brief Saturation thresholding function, to be combined with texture segmentation
 * @return cv::Mat Mask of the saturation thresholded image
 */
cv::Mat Mask::saturationThresholding()
{
	// Variables
	const int THRESHOLD = 32;

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
	cv::normalize(saturation, saturation, 0, 255, cv::NORM_MINMAX);
	cv::threshold(saturation, saturation, THRESHOLD, 255, cv::THRESH_BINARY);

	return saturation;
}

cv::Mat Mask::textureSegmentation()
{
	// Variables
	const unsigned int HISTOGRAM_DIMENSION = 9;
	const unsigned int HISTOGRAM_EDGE_OFFSET = (HISTOGRAM_DIMENSION - 1) / 2;
	const float TEXTURE_EDGE_THRESHOLD = 204;
	typedef std::map<unsigned int, int> HDM;

	// Grayscale conversion
	cv::Mat gs_image;
	cv::cvtColor(source_image, gs_image, cv::COLOR_BGR2GRAY);

	// Image padding
	auto padImage = [](const cv::Mat& input, cv::Mat& output, const unsigned int padding) -> void
	{
		auto symmetricPoint = [](const cv::Point& inP, cv::Point& outP, const cv::Size& sz, const unsigned int padding) -> void
		{
			if (inP.y < padding) outP.y = 2 * padding - inP.y - 1;
			else if (inP.y > sz.width - padding - 1) outP.y = sz.width - 2 * padding + sz.width - inP.y - 1;
			else outP.y = inP.y;

			if (inP.x < padding) outP.x = 2 * padding - inP.x - 1;
			else if (inP.x > sz.height - padding - 1) outP.x = sz.height - 2 * padding + sz.height - inP.x - 1;
			else outP.x = inP.x;
		};
		
		cv::copyMakeBorder(input, output, padding, padding, padding, padding, 0);
		cv::MatIterator_<uchar> its;
		cv::MatIterator_<uchar> it;
		cv::Size psize = input.size();

		for (unsigned int j = 0; j < psize.width; j++)
		{
			for (unsigned int i = 0; i < padding; i++)
			{
				cv::Point sp;
				cv::Point p(i, j);
				symmetricPoint(p, sp, psize, padding);
				its = output.begin<uchar>() + sp.x * output.cols + sp.y;
				it = output.begin<uchar>() + p.x * output.cols + p.y;
				*it = *its;
			}
			for (unsigned int i = psize.height - padding; i < psize.height; i++)
			{
				cv::Point sp;
				cv::Point p(i, j);
				symmetricPoint(p, sp, psize, padding);
				its = output.begin<uchar>() + sp.x * output.cols + sp.y;
				it = output.begin<uchar>() + p.x * output.cols + p.y;
				*it = *its;
			}
		}
	};
	cv::Mat padded_image;
	padImage(gs_image, padded_image, HISTOGRAM_EDGE_OFFSET);

	// Entropy matrix
	auto entropyFilter = [](const cv::Mat& input, cv::Mat& output, const unsigned int padding, const unsigned int dimension) -> void
	{
		auto initialHistogram = [](const cv::Mat& input, cv::Mat& hist, const unsigned int padding, const unsigned int dimension) -> void
		{
			cv::Size psize = input.size();
			int hsize = 256;
			float range[] = { 0, 256 };
			const float* hrange = { range };
			int channels = 0;
			bool uniform = true;
			bool accumulate = false;

			cv::Mat initial_mask = cv::Mat::ones(dimension, dimension, CV_8UC1);
			cv::copyMakeBorder(initial_mask, initial_mask, 0, psize.height - dimension, 0, psize.width - dimension, cv::BORDER_CONSTANT, 0);
			cv::calcHist(&input, 1, &channels, initial_mask, hist, 1, &hsize, &hrange, uniform, accumulate);
		};
		auto entropy = [](const cv::Mat& hist) -> double
		{
			double e = 0;
			cv::MatConstIterator_<float> it;

			for (it = hist.begin<float>(); it != hist.end<float>(); it++)
				if (*it > 0)
					e -= *it * log(*it);
			return e / 255;
		};
		auto entropyAndUpdateHistogram = [](HDM& m, double previous_pixel_entropy, cv::Mat& hist) -> double
		{
			double pixel_entropy = previous_pixel_entropy;
			cv::MatIterator_<float> hbegin = hist.begin<float>();

			for (HDM::const_iterator it = m.begin(); it != m.end(); it++)
				if (it->second != 0)
				{
					float histv = *(hbegin + (int)it->first);
					if (histv > 0)
						pixel_entropy += histv * std::log2(histv) / 255;
					histv += it->second;
					*(hbegin + (int)it->first) = histv;
					if (histv > 0)
						pixel_entropy -= histv * std::log2(histv) / 255;
				}
			m.clear();
			return pixel_entropy;
		};
		auto removeColumn = [](const cv::Mat& input, cv::Mat& hist, const cv::Point& p, HDM& m, const unsigned int dimension) -> void
		{
			cv::MatConstIterator_<uchar> it;
			cv::MatConstIterator_<uchar> begin = input.begin<uchar>() + p.x * input.size().width + p.y;
			cv::MatConstIterator_<uchar> end = begin + dimension * input.cols;
			HDM::iterator itHDM;
			
			for (it = begin; it != end; it += input.cols)
			{
				itHDM = m.find((uchar)*it);
				if (itHDM != m.end())
					itHDM->second--;
				else
					m.insert(std::pair<unsigned int, int>(*it, -1));
			}
		};
		auto addColumn = [](const cv::Mat& input, cv::Mat& hist, const cv::Point& p, HDM& m, const unsigned int dimension) -> void
		{
			cv::MatConstIterator_<uchar> it;
			cv::MatConstIterator_<uchar> begin = input.begin<uchar>() + p.x * input.size().width + p.y;
			cv::MatConstIterator_<uchar> end = begin + dimension * input.cols;
			HDM::iterator itHDM;

			for (it = begin; it != end; it += input.cols)
			{
				itHDM = m.find((uchar)*it);
				if (itHDM != m.end())
					itHDM->second++;
				else
					m.insert(std::pair<unsigned int, int>(*it, 1));
			}
		};
		auto removeRow = [](const cv::Mat& input, cv::Mat& hist, const cv::Point& p, HDM& m, const unsigned int dimension) -> void
		{
			cv::MatConstIterator_<uchar> it;
			cv::MatConstIterator_<uchar> begin = input.begin<uchar>() + p.x * input.size().width + p.y;
			cv::MatConstIterator_<uchar> end = begin + dimension;
			HDM::iterator itHDM;

			for (it = begin; it != end; it++)
			{
				itHDM = m.find((uchar)*it);
				if (itHDM != m.end())
					itHDM->second--;
				else
					m.insert(std::pair<unsigned int, int>(*it, -1));
			}
		};
		auto addRow = [](const cv::Mat& input, cv::Mat& hist, const cv::Point& p, HDM& m, const unsigned int dimension) -> void
		{
			cv::MatConstIterator_<uchar> it;
			cv::MatConstIterator_<uchar> begin = input.begin<uchar>() + p.x * input.size().width + p.y;
			cv::MatConstIterator_<uchar> end = begin + dimension;
			HDM::iterator itHDM;

			for (it = begin; it != end; it++)
			{
				itHDM = m.find((uchar)*it);
				if (itHDM != m.end())
					itHDM->second++;
				else
					m.insert(std::pair<unsigned int, int>(*it, 1));
			}
		};

		double pixel_entropy;
		const cv::Point OFFSET_POINT = cv::Point(padding, padding);
		cv::Size psize = input.size();
		cv::MatIterator_<float> ite = output.begin<float>();
		cv::Mat hist;
		initialHistogram(input, hist, padding, dimension);
		int direction = 1;
		cv::Point current_point(padding, padding);
		pixel_entropy = entropy(hist);
		*ite = pixel_entropy;
		cv::Point previous_point = current_point;
		current_point.y++;
		HDM m;

		bool stop = false;
		while (!stop)
		{
			// 1. Get current pointer
			ite = output.begin<float>() + (current_point - OFFSET_POINT).x * output.size().width + (current_point - OFFSET_POINT).y;
			
			// 2. Update histogram
			if (previous_point.y != current_point.y)
			{
				// 2.1. Remove previous column
				removeColumn(input, hist, cv::Point(previous_point.x - padding, previous_point.y - direction * padding), m, dimension);
				// 2.2. Add current column
				addColumn(input, hist, cv::Point(current_point.x - padding, current_point.y + direction * padding), m, dimension);
			}
			if (previous_point.x != current_point.x)
			{
				// 2.3. Remove previous row
				removeRow(input, hist, cv::Point(previous_point.x - padding, previous_point.y - padding), m, dimension);
				// 2.4. Add current row
				addRow(input, hist, cv::Point(current_point.x + padding, current_point.y - padding), m, dimension);
			}
			pixel_entropy = entropyAndUpdateHistogram(m, pixel_entropy, hist);
			*ite = pixel_entropy;
			
			// 3. Advance pointer
			previous_point = current_point;
			current_point.y += direction;
			if (current_point.y == psize.width - padding)
			{
				current_point = cv::Point(current_point.x + 1, current_point.y - 1);
				direction = -1;
			}
			else if (current_point.y == padding - 1)
			{
				current_point = cv::Point(current_point.x + 1, padding);
				direction = 1;
			}
			if (current_point.x == psize.height - padding)
				stop = true;
		}
	};
	cv::Mat entropy_matrix = cv::Mat::zeros(gs_image.rows, gs_image.cols, CV_32FC1);
	entropyFilter(padded_image, entropy_matrix, HISTOGRAM_EDGE_OFFSET, HISTOGRAM_DIMENSION);

	// Rescale entropy matrix
	cv::normalize(entropy_matrix, entropy_matrix, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

	// [0, 255] conversion
	auto convertMatrix = [](const cv::Mat& input, cv::Mat& output) -> void
	{
		double min, max;
		cv::minMaxLoc(input, &min, &max);
		input.convertTo(output, CV_8UC1, 255.0 / (max - min), -min * 255.0 / (max - min));
	};
	cv::Mat converted_entropy_matrix;
	convertMatrix(entropy_matrix, converted_entropy_matrix);

	// Texture thresholding
	cv::threshold(converted_entropy_matrix, converted_entropy_matrix, TEXTURE_EDGE_THRESHOLD, 255, cv::THRESH_BINARY);

	return converted_entropy_matrix;
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
*/