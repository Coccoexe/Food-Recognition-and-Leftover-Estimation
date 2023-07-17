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
#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include <Python.h>

#define DEBUG false

using namespace std;
using namespace cv;

void display(Mat img)
{
	namedWindow("Display window", WINDOW_AUTOSIZE);
	imshow("Display window", img);
	waitKey(0);
}

void k_means(Mat img, Mat out, int k)
{
	Mat samples(img.rows * img.cols, 3, CV_32F);
	for (int y = 0; y < img.rows; y++)
		for (int x = 0; x < img.cols; x++)
			for (int z = 0; z < 3; z++)
				samples.at<float>(y + x * img.rows, z) = img.at<Vec3b>(y, x)[z];

	Mat labels;
	int attempts = 5;
	Mat centers;
	kmeans(samples, k, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);

	for (int y = 0; y < img.rows; y++)
		for (int x = 0; x < img.cols; x++)
		{
			int cluster_idx = labels.at<int>(y + x * img.rows, 0);
			out.at<Vec3b>(y, x)[0] = centers.at<float>(cluster_idx, 0);
			out.at<Vec3b>(y, x)[1] = centers.at<float>(cluster_idx, 1);
			out.at<Vec3b>(y, x)[2] = centers.at<float>(cluster_idx, 2);
		}
}

cv::Mat process(cv::Mat msk1, cv::Mat imaaasss) {
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
	cv::medianBlur(msk1, msk1, 5);

	//closing
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(40, 40));
	cv::morphologyEx(msk1, msk1, cv::MORPH_CLOSE, kernel);

	//dilation
	cv::Mat a = cv::Mat::zeros(msk1.size(), CV_8UC1);
	filterAreas(msk1, a, 8000);
	cv::dilate(a, a, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15)));

	//closing
	kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15));
	cv::morphologyEx(a, a, cv::MORPH_CLOSE, kernel);

	//filling holes
	fillHoles(a);

	//detected
	cv::Mat original;
	cv::bitwise_and(imaaasss, imaaasss, original, a);
	//cv::imshow("original", original);
	//cv::waitKey(0);

	return a;
}

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
	const string PLATES_PATH = "./plates/";
	const string LABELS_PATH = "./labels/";

	// Process
	if (!filesystem::exists(PLATES_PATH)) filesystem::create_directory(PLATES_PATH);
	for (int i = 1; i <= NUMBER_OF_TRAYS; i++)
	{	// For each tray
		if (!filesystem::exists(PLATES_PATH + "tray" + to_string(i) + " / ")) filesystem::create_directory(PLATES_PATH + "tray" + to_string(i) + " / ");
		for (const auto& imgname : IMAGE_NAMES)
		{	// For each image
			if (!filesystem::exists(PLATES_PATH + "tray" + to_string(i) + "/" + imgname + "/")) filesystem::create_directory(PLATES_PATH + "tray" + to_string(i) + "/" + imgname + "/");
			cv::Mat image = cv::imread(DATASET_PATH + "tray" + to_string(i) + "/" + imgname + ".jpg");
			BoundingBoxes bb(image);
			vector<cv::Vec3f> plates = bb.getPlates();
			pair<bool, cv::Vec3f> salad = bb.getSalad();
			pair<bool, cv::Rect> bread = bb.getBread();

			// Save plates cutouts
			for (int j = 0; j < plates.size(); j++)
			{
				cv::imwrite(PLATES_PATH + "tray" + to_string(i) + "/" + imgname + "/plate" + to_string(j) + ".jpg", cutout(image, plates[j]));
			}
		}

		if (DEBUG) cout << "Running Python script..." << endl;

		//Python detect
		Py_Initialize();
		PyRun_SimpleString("import sys");
		PyRun_SimpleString("sys.path.append('../../../src/Python/')");
		PyRun_SimpleString("sys.argv = ['CLIP_interface.py']");
		PyObject* pName = PyUnicode_FromString("CLIP_interface");
		PyObject* pModule = PyImport_Import(pName);
		PyObject* pFunc = PyObject_GetAttrString(pModule, "main");
		//PyObject_CallObject(pFunc, NULL);
		
		//call method with i
		PyObject* pArgs = PyTuple_New(1);
		PyObject* pValue = PyLong_FromLong(i);
		PyTuple_SetItem(pArgs, 0, pValue);
		PyObject_CallObject(pFunc, pArgs);

		Py_Finalize();

		if (DEBUG) cout << "Python script finished" << endl;

		// Segmentation
		for (const auto& imgname : IMAGE_NAMES)
		{
			vector<string> files;
			glob(PLATES_PATH + "tray" + to_string(i) + "/" + imgname + "/*.jpg", files);
			for (const auto& file : files)
			{
				string name = file.substr(PLATES_PATH.length(), file.length()-1);
				//read label from file
				ifstream infile(LABELS_PATH + name + ".txt");
				string line;
				vector<string> labels;
				while (getline(infile, line))
				{
					labels.push_back(line);
				}
				infile.close();

				for (const auto& label : labels)
				{
					if (true) cout << "Looking for label: " << label << " in file: " << PLATES_PATH + name << endl;
					// segmentation with colors
				}

				
			}
		}

		//black image 20x20
		Mat a = Mat::zeros(20, 20, CV_8UC1);
		display(a);

	}

	return 0;
}
