// Alessio Cocco 2087635, Andrea Valentinuzzi 2090451, Giovanni Brejc
// Computer Vision final project 2022/2023 University of Padua
// Food Recognition and Leftover Estimation


#include "BoundingBox.hpp"
#include "Mask.hpp"
#include "Metrics.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

using namespace std;

int main()
{
	// 1. recognize and localize all the food items in the tray images, considering the food categories detailed in the dataset
	// 2. segment each food item in the tray image to compute the corresponding food quantity (i.e., amount of pixels)
	// 3. compare the “before meal” and “after meal” images to find which food among the initial ones was eaten and which was not. The leftovers quantity is then estimated as the difference in the number of pixels of the food item in the pair of images.

	const vector<string> labels = {
		"Background",
		"pasta with pesto",
		"pasta with tomato sauce",
		"pasta with meat sauce",
		"pasta with clams and mussels",
		"pilaw rice with peppers and peas",
		"grilled pork cutlet",
		"fish cutlet",
		"rabbit",
		"seafood salad",
		"beans",
		"basil potatoes",
		"salad",
		"bread"
	};
	vector<Metrics> tt; 																		  // Vector of Metrics objects
	string dataset = "./Food_leftover_dataset/";                                                  // Dataset folder name
	string matching = "./matching/";                                                              // Matching folder name
	string output = "./output/";                                                                  // Output folder name

	std::multimap<int, const cv::Mat> match;                                                      // Map of matching images
	for (int i = 1; i < labels.size(); i++)
	{  // Process each LABEL of images that will be used for matching
		string folder = matching + labels.at(i) + "/";                                            // Matching folder name

		for (const auto& entry : filesystem::directory_iterator(folder))
		{  // Process each IMAGE in the folder
			const cv::Mat img = cv::imread(entry.path().string());                                // Read image
			match.insert(std::pair<int, const cv::Mat>(i, img));                                  // Add (label, image) to map
		}
	}
	if (!filesystem::exists(output)) filesystem::create_directory(output);                        // Create output folder if it doesn't exist
	for (int tray = 1; tray <= 8; tray++)
	{	// Process each TRAY set of images
		string folder = dataset + "tray" + to_string(tray) + "/";                                 // Tray folder name
		string files[] = { "food_image.jpg", "leftover1.jpg", "leftover2.jpg", "leftover3.jpg" }; // Image file names
		string bounding_boxes = folder + "bounding_boxes/";                                       // bounding_boxes folder
		string masks = folder + "masks/";                                                         // masks folder
		vector<BoundingBox> bb;                                                                   // Vector of BoundingBox objects
		vector<Mask> mm;                                                                          // Vector of Mask objects

		if (!filesystem::exists(bounding_boxes)) filesystem::create_directory(bounding_boxes);    // Create bounding_boxes folder if it doesn't exist
		if (!filesystem::exists(masks)) filesystem::create_directory(masks);                      // Create masks folder if it doesn't exist
		for (int image = 0; image < 4; image++)
		{	// Process each IMAGE in the tray
			const cv::Mat img = cv::imread(folder + files[image]);                                // Read image

			bb.push_back(BoundingBox(img, labels, match));                                        // Create BoundingBox object
			mm.push_back(Mask(img, bb.at(image)));                                                // Create Mask object
		}
		tt.push_back(Metrics(bb, mm)); 															  // Create Metrics object
	}


	return 0;
}