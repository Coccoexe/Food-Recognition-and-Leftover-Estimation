// Alessio Cocco 2087635, Andrea Valentinuzzi 2090451, Giovanni Brejc
// Computer Vision final project 2022/2023 University of Padua
// Food Recognition and Leftover Estimation
//
// 1. Detect plates            --> OpenAI CLIP to find k classes per plate --> k+1 means segmentation
// 2. Detect salad (if exists) --> grabCut segmentation
// 3. Detect bread (if exists) --> grabCut segmentation

#include "BoundingBoxes.hpp"

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#define DEBUG true

using namespace std;

int main()
{
	// Variables
	const string DATASET_PATH = "./Food_leftover_dataset/";
	const int NUMBER_OF_TRAYS = 8;
	const vector<string> IMAGE_NAMES = { "food_image", "leftover1", "leftover2", "leftover3" };

	// Cycle through all images
	for (int i = 1; i <= NUMBER_OF_TRAYS; i++)
	{	// For each tray
		for (const auto& imgname : IMAGE_NAMES)
		{	// For each image
			cv::Mat image = cv::imread(DATASET_PATH + "tray" + to_string(i) + "/" + imgname + ".jpg");
			BoundingBoxes bb(image);
		}
	}

	return 0;
}
