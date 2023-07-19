#include "Metrics.hpp"

#define DEBUG false

Metrics::Metrics(std::vector<std::vector<std::tuple<cv::Mat, std::vector<std::pair<int, cv::Rect>>, cv::Mat, std::vector<std::pair<int, cv::Rect>>>>> &m)
	: metrics(m)
{
	true_positives = std::vector<int>(14, 0);
	false_positives = std::vector<int>(14, 0);
	false_negatives = std::vector<int>(14, 0);
	precision = std::vector<std::vector<double>>(14, std::vector<double>());
	recall = std::vector<std::vector<double>>(14, std::vector<double>());

	for (const auto tray : metrics) // for each tray
	{
		for (const auto img : tray) // for each image in the tray
		{
			cv::Mat mask = std::get<0>(img);											// mask computed by segmentation
			cv::Mat orig_mask = std::get<2>(img);										// mask computed by ground truth
			std::vector<std::pair<int, cv::Rect>> labeled_box = std::get<1>(img);		// labels computed by segmentation
			std::vector<std::pair<int, cv::Rect>> orig_labeled_box = std::get<3>(img);	// labels computed by ground truth

			// compute precision and recall for each label
			for (int i = 0; i < orig_labeled_box.size(); i++)
			{
				int orig_label = orig_labeled_box[i].first;
				cv::Rect orig_rect = orig_labeled_box[i].second;
				// if label is not present in computed labels
				if (std::find_if(labeled_box.begin(), labeled_box.end(), [orig_label](const std::pair<int, cv::Rect>& p) { return p.first == orig_label; }) == labeled_box.end())
				{
					false_negatives[orig_label]++;
					if ((true_positives[orig_label] + false_positives[orig_label]) != 0)
						precision[orig_label].push_back(true_positives[orig_label] / (true_positives[orig_label] + false_positives[orig_label]));
					else
						precision[orig_label].push_back(0);
					recall[orig_label].push_back(true_positives[orig_label] / (true_positives[orig_label] + false_negatives[orig_label]));
					continue;
				}
				//compute intersection over union
				cv::Rect label_rect = std::find_if(labeled_box.begin(), labeled_box.end(), [orig_label](const std::pair<int, cv::Rect>& p) { return p.first == orig_label; })->second;
				cv::Rect intersection = label_rect & orig_rect;
				cv::Rect union_ = label_rect | orig_rect;
				double iou = (double)intersection.area() / (double)union_.area();

				std::cout << "iou: " << iou << std::endl;
					//print the rectangles
				cv::Mat temp = orig_mask.clone();
				cv::rectangle(temp, label_rect, cv::Scalar(0, 255, 0), 2);
				cv::rectangle(temp, orig_rect, cv::Scalar(0, 0, 255), 2);
				cv::imshow("rectangles", temp);
				cv::waitKey(0);
				
				if (iou > 0.5) true_positives[orig_label]++;
				else if (iou > 0) false_positives[orig_label]++;
				else false_negatives[orig_label]++;

				if ((true_positives[orig_label] + false_positives[orig_label]) != 0)
					precision[orig_label].push_back(true_positives[orig_label] / (true_positives[orig_label] + false_positives[orig_label]));
				else
					precision[orig_label].push_back(0);

				if ((true_positives[orig_label] + false_negatives[orig_label]) != 0)
					recall[orig_label].push_back(true_positives[orig_label] / (true_positives[orig_label] + false_negatives[orig_label]));
				else
					recall[orig_label].push_back(0);
			}
		}
	}

	
	int gne = 7;

	std::cout << "TP: " << true_positives[gne] << " FP: " << false_positives[gne] << " FN: " << false_negatives[gne] << std::endl;

	for (int i = 0; i < precision[gne].size(); i++)
	{
		std::cout << "precision: " << precision[gne][i] << " recall: " << recall[gne][i] << std::endl;
	}


	for (int i = 0; i < 14; i++)
	{
		double sum = 0;
		for (int j = 0; j < precision[i].size(); j++)
		{
			sum += precision[i][j];
		}
		average_precision.push_back(sum / precision[i].size());
		std::cout << average_precision[i] << std::endl;
	}

	cv::imshow("mask", std::get<0>(metrics[0][0]));
	cv::waitKey(0);

}
