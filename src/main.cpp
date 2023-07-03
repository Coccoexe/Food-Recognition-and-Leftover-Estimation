// main.cpp : Defines the entry point for the application.
//

#include "main.hpp"
#include <opencv2/opencv.hpp>
#include <string>
#include <filesystem>

using namespace std;

int main()
{
	// 1. recognize and localize all the food items in the tray images, considering the food categories detailed in the dataset
	// 2. segment each food item in the tray image to compute the corresponding food quantity (i.e., amount of pixels)
	// 3. compare the “before meal” and “after meal” images to find which food among the initial ones was eaten and which was not. The leftovers quantity is then estimated as the difference in the number of pixels of the food item in the pair of images.

	const int num_labels = 14;                                                                    // Number of labels
	string labels[num_labels] = {                                                                 // Labels
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
	string dataset = "./Food_leftover_dataset/";                                                  // Dataset folder name
	string output = "./output/";                                                                  // Output folder name
	if (!filesystem::exists(output)) filesystem::create_directory(output);                        // Create output folder if it doesn't exist
	for (int tray = 1; tray <= 8; tray++)
	{	// Process each TRAY set of images
		string folder = dataset + "tray" + to_string(tray) + "/";                                 // Tray folder name
		string files[] = { "food_image.jpg", "leftover1.jpg", "leftover2.jpg", "leftover3.jpg" }; // Image file names
		string bounding_boxes = folder + "bounding_boxes/";                                       // bounding_boxes folder
		if (!filesystem::exists(bounding_boxes)) filesystem::create_directory(bounding_boxes);    // Create bounding_boxes folder if it doesn't exist
		string masks = folder + "masks/";                                                         // masks folder
		if (!filesystem::exists(masks)) filesystem::create_directory(masks);                      // Create masks folder if it doesn't exist
		for (int image = 0; image < 4; image++)
		{	// Process each IMAGE in the tray
			cv::Mat img = cv::imread(folder + files[image]);                                      // Read image
			cv::imshow("Original", img);                                                          // Show original image
			cv::waitKey(0);
			
		}
	}


	return 0;
}
