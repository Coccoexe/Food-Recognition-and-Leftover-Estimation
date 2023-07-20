#include "Metrics.hpp"
#include <iostream>
#include <fstream>

#define DEBUG false


Metrics::Metrics(std::vector<std::vector<std::tuple<cv::Mat, std::vector<std::pair<int, cv::Rect>>, cv::Mat, std::vector<std::pair<int, cv::Rect>>>>>& m)
	: metrics(m)
{
	true_positives = std::vector<double>(14, 0);							 // true positives for each class							
	false_positives = std::vector<double>(14, 0);							 // false positives for each class
	false_negatives = std::vector<double>(14, 0);							 // false negatives for each class
	precision = std::vector<std::vector<double>>(14, std::vector<double>()); // precision for each class
	recall = std::vector<std::vector<double>>(14, std::vector<double>());	 // recall for each class
	IoU = std::vector<std::vector<double>>(14, std::vector<double>());		 // IoU for each class
	average_precision = std::vector<double>(14, 0);							 // average precision for each class

	// mIoU
	for (const auto& tray : metrics)
	{ // for each tray
		for (int i = 0; i < 3; i++)
		{ // for each image in the tray except for leftover 3
			cv::Mat mask = std::get<0>(tray[i]);											// mask computed by segmentation
			cv::Mat orig_mask = std::get<2>(tray[i]);										// mask computed by ground truth
			std::vector<std::pair<int, cv::Rect>> labeled_box = std::get<1>(tray[i]);		// labels computed by segmentation
			std::vector<std::pair<int, cv::Rect>> orig_labeled_box = std::get<3>(tray[i]);	// labels computed by ground truth

			for (const auto& olb : orig_labeled_box)
			{
				cv::Mat thresh_mask, thres_orig_mask;
				cv::compare(mask, olb.first, thresh_mask, cv::CMP_EQ);
				cv::compare(orig_mask, olb.first, thres_orig_mask, cv::CMP_EQ);
				cv::Mat intersection = thresh_mask & thres_orig_mask;
				cv::Mat union_ = thresh_mask | thres_orig_mask;

				double iou = (double)cv::countNonZero(intersection) / (double)cv::countNonZero(union_);
				IoU[olb.first].push_back(iou);
			}
		}
	}
	for (int i = 0; i < 14; i++)
	{
		double sum = 0;
		for (const auto& iou : IoU[i])
			sum += iou;
		std::cout << "IoU[" << i << "] = " << sum / IoU[i].size() << std::endl;
	}


	// Food leftover estimation
	std::ofstream file;
	file.open("./output/Food_leftover.txt");
	file << "Food leftover estimation" << std::endl;
	file.close();

	for (int i = 0; i < metrics.size(); i++)
	{
		//compute pixels of food in the food_tray
		cv::Mat mask = std::get<0>(metrics[i][0]);												// mask computed by segmentation
		cv::Mat orig_mask = std::get<2>(metrics[i][0]);											// mask computed by ground truth
		std::vector<std::pair<int, cv::Rect>> labeled_box = std::get<1>(metrics[i][0]);			// labels computed by segmentation
		std::vector<std::pair<int, cv::Rect>> orig_labeled_box = std::get<3>(metrics[i][0]);	// labels computed by ground truth

		std::vector<std::string> food_leftover;

		if (DEBUG) std::cout << "Tray " << i << std::endl;

		for (int j = 1; j < 4; j++)
		{
			if (DEBUG) std::cout << "   Leftover " << j << std::endl;

			cv::Mat mask_left = std::get<0>(metrics[i][j]);											// mask computed by segmentation
			cv::Mat orig_mask_left = std::get<2>(metrics[i][j]);										// mask computed by ground truth

			for (const auto& olb : orig_labeled_box)
			{

				double food_pixels, orig_food_pixels;
				double food_pixels_left, orig_food_pixels_left;

				// pixels of food in the food_tray
				cv::Mat thresh_mask, thres_orig_mask;
				cv::compare(mask, olb.first, thresh_mask, cv::CMP_EQ);
				cv::compare(orig_mask, olb.first, thres_orig_mask, cv::CMP_EQ);
				food_pixels = cv::countNonZero(thresh_mask);
				orig_food_pixels = cv::countNonZero(thres_orig_mask);

				// pixels of food in the leftover 3
				cv::Mat thresh_mask_left, thres_orig_mask_left;
				cv::compare(mask_left, olb.first, thresh_mask_left, cv::CMP_EQ);
				cv::compare(orig_mask_left, olb.first, thres_orig_mask_left, cv::CMP_EQ);
				food_pixels_left = cv::countNonZero(thresh_mask_left);
				orig_food_pixels_left = cv::countNonZero(thres_orig_mask_left);

				double estimated_leftover = (food_pixels_left / food_pixels);
				double actual_leftover = (orig_food_pixels_left / orig_food_pixels);
				if (DEBUG)
				{
					std::cout << "Estimated leftover of food " << olb.first << " = " << estimated_leftover << std::endl;
					std::cout << "Actual leftover of food " << olb.first << " = " << actual_leftover << std::endl;
					std::cout << "Difference = " << abs(estimated_leftover - actual_leftover) << std::endl;
				}
				std::string temp = "      Real Leftover: " + std::to_string(actual_leftover) + "\n" +
					"      Estimated Leftover: " + std::to_string(estimated_leftover) + "\n" +
					"      Difference: " + std::to_string(abs(estimated_leftover - actual_leftover));
				food_leftover.push_back(temp);
			}
		}

		//write on file
		std::ofstream file;
		file.open("./output/Food_leftover.txt", std::ios_base::app);
		file << "Tray " << i << std::endl;
		for (int j = 0; j < food_leftover.size(); j++)
		{
			file << "Leftover " << j + 1 << std::endl;
			file << food_leftover[j] << std::endl;
		}
		file << std::endl;
	}


	// mAP
	std::vector<double> occurrency = std::vector<double>(14, 0);
	for (const auto& tray : metrics)
	{
		for (int i = 0; i < 3; i++)
		{
			std::vector<std::pair<int, cv::Rect>> orig_labeled_box = std::get<3>(tray[i]);	// labels computed by ground truth
			for (const auto& olb : orig_labeled_box)
				occurrency[olb.first]++;
		}
	}
	for (const auto& tray : metrics) // for each tray
	{
		for (int i = 0; i < 3; i++) // for each image in the tray except for leftover 3
		{
			cv::Mat mask = std::get<0>(tray[i]);											// mask computed by segmentation
			cv::Mat orig_mask = std::get<2>(tray[i]);										// mask computed by ground truth
			std::vector<std::pair<int, cv::Rect>> labeled_box = std::get<1>(tray[i]);		// labels computed by segmentation
			std::vector<std::pair<int, cv::Rect>> orig_labeled_box = std::get<3>(tray[i]);	// labels computed by ground truth

			for (const auto& olb : orig_labeled_box)
			{
				// find the label in the computed labels
				auto lb = std::find_if(labeled_box.begin(), labeled_box.end(), [olb](const std::pair<int, cv::Rect>& p) { return p.first == olb.first; });
				if (lb == labeled_box.end())
				{// if label is not present in computed labels
					false_negatives[olb.first]++; // increment false negatives
					continue;
				}

				cv::Rect intersection = lb->second & olb.second;					// compute intersection
				cv::Rect union_ = lb->second | olb.second;							// compute union
				double iou = (double)intersection.area() / (double)union_.area();	// compute intersection over union

				if (iou >= 0.5)					   // iou threshold
					true_positives[olb.first]++;   // increment true positives
				else
					false_positives[olb.first]++;  // increment false negatives

				labeled_box.erase(lb); // remove the label from the computed labels


			}

			for (const auto& lb : labeled_box)
				false_positives[lb.first]++; // add false positives for each label


			// compute precision and recall for each label
			for (int i = 0; i < orig_labeled_box.size(); i++)
			{
				int orig_label = orig_labeled_box[i].first;
				if ((true_positives[orig_label] + false_positives[orig_label]) != 0)
					precision[orig_label].push_back(true_positives[orig_label] / (true_positives[orig_label] + false_positives[orig_label]));
				else
					precision[orig_label].push_back(0);
				recall[orig_label].push_back(true_positives[orig_label] / occurrency[orig_label]);//(true_positives[orig_label] + false_negatives[orig_label]));

			}
		}
	}
	if (DEBUG)
	{
		//print all precision and recall values
		for (int i = 1; i < 14; i++)
		{
			std::cout << "label: " << i << std::endl;
			std::cout << "true positives: " << true_positives[i] << std::endl;
			std::cout << "false positives: " << false_positives[i] << std::endl;
			std::cout << "false negatives: " << false_negatives[i] << std::endl;

			for (int j = 0; j < precision[i].size(); j++)
			{
				std::cout << "precision: " << precision[i][j] << " recall: " << recall[i][j] << std::endl;
			}
		}
	}
	for (int i = 1; i < 14; i++)
	{
		double ap = 0.0;
		for (int j = 0; j < 11; j++)
		{
			double thresh = (double)j / 10.0;
			double max = 0.0;
			for (int k = 0; k < recall[i].size(); k++)
			{
				if (recall[i][k] < thresh) continue;
				if (precision[i][k] > max) max = precision[i][k];
			}
			ap += max;
		}
		if (DEBUG) std::cout << "label: " << i << " ap: " << ap / 11.0 << std::endl;
		average_precision[i] = ap / 11.0;
	}
	//mean
	double mean = 0.0;
	for (int i = 1; i < 14; i++)
		mean += average_precision[i];
	mean /= 13.0;
	std::cout << "mean: " << mean << std::endl;

	//write results to file
	file.open("./output/metrics_results.txt");

	//mAP
	file << "Average Precision for each class: " << std::endl;
	for (int i = 1; i < 14; i++)
		file << "label: " << i << " AP: " << average_precision[i] << std::endl;
	file << std::endl;
	file << "Mean Average Precision: " << mean << std::endl << std::endl;

	//mIoU
	file << "Mean Intersection over Union for each class: " << std::endl;
	for (int i = 1; i < 14; i++)
	{
		double mean_iou = 0.0;
		for (int j = 0; j < IoU[i].size(); j++)
			mean_iou += IoU[i][j];
		mean_iou /= IoU[i].size();
		file << "label: " << i << " mIoU: " << mean_iou << std::endl;
	}
	file << std::endl;
	file.close();
}