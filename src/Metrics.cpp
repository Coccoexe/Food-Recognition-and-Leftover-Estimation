#include "Metrics.hpp"

#include <iostream>
#include <fstream>

#define DEBUG false

Metrics::Metrics(std::vector<std::vector<std::tuple<cv::Mat, std::vector<std::pair<int, cv::Rect>>, cv::Mat, std::vector<std::pair<int, cv::Rect>>>>>& m)
	: metrics(m)
{
	true_positives = std::vector<double>(14, 0);							   // TP: True positives for each class							
	false_positives = std::vector<double>(14, 0);							   // FP: False positives for each class
	false_negatives = std::vector<double>(14, 0);							   // FN: False negatives for each class
	precision = std::vector<std::vector<double>>(14, std::vector<double>());   // Precision for each class
	recall = std::vector<std::vector<double>>(14, std::vector<double>());	   // Recall for each class
	IoU = std::vector<std::vector<double>>(14, std::vector<double>());		   // IoU for each class
	average_precision = std::vector<double>(14, 0);							   // Average precision for each class

	// Compute mIoU
	for (const auto& tray : metrics)
	{	// For each 'tray' in the metrics vector
		for (int i = 0; i < 3; i++)
		{	// For each image [i] in the tray, except for leftover3
			cv::Mat mask = std::get<0>(tray[i]);										     // Mask computed by segmentation
			cv::Mat orig_mask = std::get<2>(tray[i]);									     // Mask computed by ground truth
			std::vector<std::pair<int, cv::Rect>> labeled_box = std::get<1>(tray[i]);	     // Labels computed by segmentation
			std::vector<std::pair<int, cv::Rect>> orig_labeled_box = std::get<3>(tray[i]);   // Labels computed by ground truth

			for (const auto& olb : orig_labeled_box)
			{	// For each label 'olb' in the ground truth
				cv::Mat thresh_mask, thres_orig_mask;
				cv::compare(mask, olb.first, thresh_mask, cv::CMP_EQ);
				cv::compare(orig_mask, olb.first, thres_orig_mask, cv::CMP_EQ);
				cv::Mat intersection = thresh_mask & thres_orig_mask;
				cv::Mat union_ = thresh_mask | thres_orig_mask;

				// Compute IoU
				double iou = (double)cv::countNonZero(intersection) / (double)cv::countNonZero(union_);
				IoU[olb.first].push_back(iou);
			}
		}
	}

	// Food leftover estimation
	std::ofstream file;
	file.open("./output/Food_leftover.txt");
	file << "Food leftover estimation" << std::endl;
	file.close();

	for (int i = 0; i < metrics.size(); i++)
	{	// For each tray [i] in the metrics vector
		cv::Mat mask = std::get<0>(metrics[i][0]);											   // Mask computed by segmentation
		cv::Mat orig_mask = std::get<2>(metrics[i][0]);										   // Mask computed by ground truth
		std::vector<std::pair<int, cv::Rect>> labeled_box = std::get<1>(metrics[i][0]);		   // Labels computed by segmentation
		std::vector<std::pair<int, cv::Rect>> orig_labeled_box = std::get<3>(metrics[i][0]);   // Labels computed by ground truth

		std::vector<std::vector<std::string>> food_leftover = std::vector<std::vector<std::string>>(3, std::vector<std::string>()); // Food leftover for each tray

		if (DEBUG) std::cout << "Tray " << i << std::endl;

		// Compute food leftover
		for (int j = 1; j < 4; j++)
		{	// For each image [j] in the tray, except for the first one
			if (DEBUG) std::cout << "   Leftover " << j << std::endl;

			cv::Mat mask_left = std::get<0>(metrics[i][j]);		   // Mask computed by segmentation
			cv::Mat orig_mask_left = std::get<2>(metrics[i][j]);   // Mask computed by ground truth

			// Count non zero pixels for each food type
			for (const auto& olb : orig_labeled_box)
			{	// For each label 'olb' in the ground truth
				double food_pixels, orig_food_pixels;
				double food_pixels_left, orig_food_pixels_left;

				// Pixels of food in the food_tray
				cv::Mat thresh_mask, thres_orig_mask;
				cv::compare(mask, olb.first, thresh_mask, cv::CMP_EQ);
				cv::compare(orig_mask, olb.first, thres_orig_mask, cv::CMP_EQ);
				food_pixels = cv::countNonZero(thresh_mask);
				orig_food_pixels = cv::countNonZero(thres_orig_mask);

				// Pixels of food in the leftover 3
				cv::Mat thresh_mask_left, thres_orig_mask_left;
				cv::compare(mask_left, olb.first, thresh_mask_left, cv::CMP_EQ);
				cv::compare(orig_mask_left, olb.first, thres_orig_mask_left, cv::CMP_EQ);
				food_pixels_left = cv::countNonZero(thresh_mask_left);
				orig_food_pixels_left = cv::countNonZero(thres_orig_mask_left);

				// Compute estimated leftover
				double estimated_leftover = (food_pixels_left / food_pixels);
				double actual_leftover = (orig_food_pixels_left / orig_food_pixels);
				if (DEBUG)
				{
					std::cout << "Estimated leftover of food " << olb.first << " = " << estimated_leftover << std::endl;
					std::cout << "Actual leftover of food " << olb.first << " = " << actual_leftover << std::endl;
					std::cout << "Difference = " << abs(estimated_leftover - actual_leftover) << std::endl;
				}

				std::string temp = "Food " + std::to_string(olb.first) + "\n" +
					"      Real Leftover: " + std::to_string(actual_leftover) + "\n" +
					"      Estimated Leftover: " + std::to_string(estimated_leftover) + "\n" +
					"      Difference: " + std::to_string(abs(estimated_leftover - actual_leftover));
				food_leftover[j - 1].push_back(temp);
			}
		}

		// Write food leftover to file
		std::ofstream file;
		file.open("./output/Food_leftover.txt", std::ios_base::app);
		file << "Tray " << i + 1 << std::endl;
		for (int j = 0; j < food_leftover.size(); j++)
		{	// For each leftover image [j] in the tray
			file << "Leftover " << j + 1 << std::endl;
			for (int k = 0; k < food_leftover[j].size(); k++)
			{	// For each food type [k] in the leftover image
				file << food_leftover[j][k] << std::endl;
			}
		}
		file << std::endl;
	}

	// Compute mAP
	std::vector<double> occurrency = std::vector<double>(14, 0);   // Number of occurrencies of each food type
	for (const auto& tray : metrics)
	{	// For each 'tray' in the metrics vector
		for (int i = 0; i < 3; i++)
		{	// For each image [i] in the tray except for leftover 3
			std::vector<std::pair<int, cv::Rect>> orig_labeled_box = std::get<3>(tray[i]);   // Labels computed by ground truth
			for (const auto& olb : orig_labeled_box)
				occurrency[olb.first]++;
		}
	}
	for (const auto& tray : metrics)
	{	// For each 'tray' in the metrics vector
		for (int i = 0; i < 3; i++)
		{	// For each image [i] in the tray except for leftover 3
			cv::Mat mask = std::get<0>(tray[i]);										     // Mask computed by segmentation
			cv::Mat orig_mask = std::get<2>(tray[i]);									     // Mask computed by ground truth
			std::vector<std::pair<int, cv::Rect>> labeled_box = std::get<1>(tray[i]);	     // Labels computed by segmentation
			std::vector<std::pair<int, cv::Rect>> orig_labeled_box = std::get<3>(tray[i]);   // Labels computed by ground truth

			for (const auto& olb : orig_labeled_box)
			{	// For each label 'olb' in the ground truth

				// Find label in computed labels
				auto lb = std::find_if(labeled_box.begin(), labeled_box.end(), [olb](const std::pair<int, cv::Rect>& p) { return p.first == olb.first; });
				if (lb == labeled_box.end())
				{	// If label is not found
					false_negatives[olb.first]++;   // Increment false negatives
					continue;
				}

				cv::Rect intersection = lb->second & olb.second;				    // Compute intersection
				cv::Rect union_ = lb->second | olb.second;						    // Compute union
				double iou = (double)intersection.area() / (double)union_.area();   // Compute intersection over union

				iou >= 0.5                            // IoU threshold
					? true_positives[olb.first]++     // Increment true positives
					: false_positives[olb.first]++;   // Increment false negatives

				labeled_box.erase(lb);   // Remove the label from the computed labels
			}

			// False positives
			for (const auto& lb : labeled_box)
				false_positives[lb.first]++; // add false positives for each label 'lb' in the computed labels

			// Compute precision and recall
			for (int i = 0; i < orig_labeled_box.size(); i++)
			{	// For each label [i] in the ground truth
				int orig_label = orig_labeled_box[i].first;   // Label of the ground truth

				(true_positives[orig_label] + false_positives[orig_label]) != 0
					? precision[orig_label].push_back(true_positives[orig_label] / (true_positives[orig_label] + false_positives[orig_label]))
					: precision[orig_label].push_back(0);

				recall[orig_label].push_back(true_positives[orig_label] / occurrency[orig_label]);   // (true_positives[orig_label] + false_negatives[orig_label]))
			}
		}
	}

	if (DEBUG)
	{	// Print all precision and recall values
		for (int i = 1; i < 14; i++)
		{	// For each label [i]
			std::cout << "label: " << i << std::endl;
			std::cout << "true positives: " << true_positives[i] << std::endl;
			std::cout << "false positives: " << false_positives[i] << std::endl;
			std::cout << "false negatives: " << false_negatives[i] << std::endl;

			for (int j = 0; j < precision[i].size(); j++)
				std::cout << "precision: " << precision[i][j] << " recall: " << recall[i][j] << std::endl;
		}
	}

	// Compute average precision
	for (int i = 1; i < 14; i++)
	{	// For each label [i]
		double ap = 0.0;   // Initialize average precision

		for (int j = 0; j < 11; j++)
		{	// For each threshold [j]
			double thresh = (double)j / 10.0;   // Compute threshold
			double max = 0.0;                   // Initialize max precision

			for (int k = 0; k < recall[i].size(); k++)
			{	// For each recall [k]
				if (recall[i][k] < thresh)
					continue;
				if (precision[i][k] > max)
					max = precision[i][k];
			}
			ap += max;   // Add max precision to average precision
		}
		if (DEBUG) std::cout << "label: " << i << " ap: " << ap / 11.0 << std::endl;

		average_precision[i] = ap / 11.0;   // Compute average precision
	}

	// Compute mAP
	double mean = 0.0;   // Initialize mean average precision
	for (int i = 1; i < 14; i++)
		mean += average_precision[i];
	mean /= 13.0;

	if (DEBUG) std::cout << "mean: " << mean << std::endl;

	// Write results to file
	file.open("./output/metrics_results.txt");

	// Write mAP
	file << "Average Precision for each class: " << std::endl;
	for (int i = 1; i < 14; i++)
		file << "label: " << i << " AP: " << average_precision[i] << std::endl;
	file << std::endl;
	file << "Mean Average Precision: " << mean << std::endl << std::endl;

	// Write mIoU
	file << "Mean Intersection over Union for each class: " << std::endl;
	for (int i = 1; i < 14; i++)
	{	// For each label [i]
		double mean_iou = 0.0;   // Initialize mean IoU

		for (int j = 0; j < IoU[i].size(); j++)
			mean_iou += IoU[i][j];
		mean_iou /= IoU[i].size();

		if (DEBUG) std::cout << "label: " << i << " mIoU: " << mean_iou << std::endl;

		file << "label: " << i << " mIoU: " << mean_iou << std::endl;
	}
	file << std::endl;

	file.close();
}