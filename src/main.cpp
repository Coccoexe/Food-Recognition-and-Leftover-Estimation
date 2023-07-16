// Alessio Cocco 2087635, Andrea Valentinuzzi 2090451, Giovanni Brejc
// Computer Vision final project 2022/2023 University of Padua
// Food Recognition and Leftover Estimation
//
// 1. Detect plates            --> OpenAI CLIP to find k classes per plate --> k+1 means segmentation
// 2. Detect salad (if exists) --> grabCut segmentation
// 3. Detect bread (if exists) --> grabCut segmentation

#include "BoundingBoxes.hpp"

#include <string>
#include <filesystem>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>


#define DEBUG true

using namespace std;

int main()
{
	// Variables
	const string DATASET_PATH = "./Food_leftover_dataset/";
	const int NUMBER_OF_TRAYS = 8;
	const vector<string> IMAGE_NAMES = { "food_image", "leftover1", "leftover2", "leftover3" };
	auto cutout = [](const cv::Mat& image, const cv::Vec3f& circle) -> cv::Mat
	{
		const int x = cvRound(circle[0] - circle[2]) > 0 ? cvRound(circle[0] - circle[2]) : 0;
		const int y = cvRound(circle[1] - circle[2]) > 0 ? cvRound(circle[1] - circle[2]) : 0;
		const int w = x + cvRound(2 * circle[2]) < image.cols ? cvRound(2 * circle[2]) : image.cols - x;
		const int h = y + cvRound(2 * circle[2]) < image.rows ? cvRound(2 * circle[2]) : image.rows - y;
		return image(cv::Rect(x, y, w, h));
	};

	// Process
	if (!filesystem::exists("./plates/")) filesystem::create_directory("./plates/");
	for (int i = 1; i <= NUMBER_OF_TRAYS; i++)
	{	// For each tray
		if (!filesystem::exists("./plates/tray" + to_string(i) + "/")) filesystem::create_directory("./plates/tray" + to_string(i) + "/");
		for (const auto& imgname : IMAGE_NAMES)
		{	// For each image
			if (!filesystem::exists("./plates/tray" + to_string(i) + "/" + imgname + "/")) filesystem::create_directory("./plates/tray" + to_string(i) + "/" + imgname + "/");
			cv::Mat image = cv::imread(DATASET_PATH + "tray" + to_string(i) + "/" + imgname + ".jpg");
			BoundingBoxes bb(image);
			vector<cv::Vec3f> plates = bb.getPlates();
			pair<bool, cv::Vec3f> salad = bb.getSalad();
			pair<bool, cv::Rect> bread = bb.getBread();

			// Save plates cutouts
			for (int j = 0; j < plates.size(); j++)
			{
				cv::imwrite("./plates/tray" + to_string(i) + "/" + imgname + "/plate" + to_string(j) + ".jpg", cutout(image, plates[j]));
				// run CLIP
				// get k classes:
				//     if k == 1 --> grabCut segmentation
				//     if k > 1  --> k+1 means segmentation
			}
		}
	}

	return 0;
}
