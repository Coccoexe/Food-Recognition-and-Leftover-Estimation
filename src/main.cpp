// Alessio Cocco 2087635, Andrea Valentinuzzi 2090451, Giovanni Brejc 2096046
// Computer Vision final project 2022/2023 University of Padua
// Food Recognition and Leftover Estimation
//
// 1. Detect plates            --> OpenAI CLIP to find k classes per plate --> k+1 means segmentation
// 2. Detect salad (if exists) --> grabCut segmentation
// 3. Detect bread (if exists) --> grabCut segmentation

#include "BoundingBoxes.hpp"
#include "Segmentation.hpp"
#include "Metrics.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <queue>
#include <string>
#include <vector>
#include <cmath>
#include <format>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include <Python.h>

#define DEBUG true
#define SKIP true	// avoid processing of CLIP

using namespace std;

void display(cv::Mat img)
{
	cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
	cv::imshow("Display window", img);
	cv::waitKey(0);
}

void k_means(cv::Mat img, cv::Mat out, int k)
{
	cv::Mat samples(img.rows * img.cols, 3, CV_32F);
	for (int y = 0; y < img.rows; y++)
		for (int x = 0; x < img.cols; x++)
			for (int z = 0; z < 3; z++)
				samples.at<float>(y + x * img.rows, z) = img.at<cv::Vec3b>(y, x)[z];

	cv::Mat labels;
	int attempts = 5;
	cv::Mat centers;
	cv::kmeans(samples, k, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10000, 0.0001), attempts, cv::KMEANS_PP_CENTERS, centers);

	for (int y = 0; y < img.rows; y++)
		for (int x = 0; x < img.cols; x++)
		{
			int cluster_idx = labels.at<int>(y + x * img.rows, 0);
			out.at<cv::Vec3b>(y, x)[0] = centers.at<float>(cluster_idx, 0);
			out.at<cv::Vec3b>(y, x)[1] = centers.at<float>(cluster_idx, 1);
			out.at<cv::Vec3b>(y, x)[2] = centers.at<float>(cluster_idx, 2);
		}
}

cv::Mat process(cv::Mat msk1) {
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

	return a;
}

int main()
{	
	// Variables
	const string DATASET_PATH = "./Food_leftover_dataset/";
	const int NUMBER_OF_TRAYS = 8;
	const vector<string> IMAGE_NAMES = { "food_image", "leftover1", "leftover2", "leftover3" };
	const string PLATES_PATH = "./plates/";
	const string BREAD_PATH = "./bread/";
	const string LABELS_PATH = "./labels/";
	const string BREAD_OUT_PATH = "./bread_output/";
	const string OUTPUT_PATH = "./output/";
	auto cutout = [](const cv::Mat& image, const cv::Vec3f& circle) -> cv::Mat
	{
		const int x = cvRound(circle[0] - circle[2]) > 0 ? cvRound(circle[0] - circle[2]) : 0;
		const int y = cvRound(circle[1] - circle[2]) > 0 ? cvRound(circle[1] - circle[2]) : 0;
		const int w = x + cvRound(2 * circle[2]) < image.cols ? cvRound(2 * circle[2]) : image.cols - x;
		const int h = y + cvRound(2 * circle[2]) < image.rows ? cvRound(2 * circle[2]) : image.rows - y;
		//std::cout << x << " " << y << " " << w << " " << h << std::endl;
		//return image(cv::Rect(x, y, w, h));
		
		//return image inside circle
		cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
		cv::circle(mask, cv::Point(cvRound(circle[0]), cvRound(circle[1])), cvRound(circle[2]), cv::Scalar(255), -1);
		cv::Mat res;
		image.copyTo(res, mask);
		return res(cv::Rect(x,y,w,h));

	};

	// Python initialization
	Py_Initialize();
	PyEval_InitThreads();
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("sys.path.append('../../../src/Python/')");
	PyRun_SimpleString("sys.argv = ['CLIP_interface.py']");
	PyObject* pName = PyUnicode_FromString("CLIP_interface");
	PyObject* pModule = PyImport_ImportModule("CLIP_interface");
	PyObject* pFunc = PyObject_GetAttrString(pModule, "plates");
	PyObject* pFunc_bread = PyObject_GetAttrString(pModule, "bread");
	PyObject* pArgs = PyTuple_New(1);

	// Process
	if (!filesystem::exists(PLATES_PATH)) filesystem::create_directory(PLATES_PATH);
	if (!filesystem::exists(OUTPUT_PATH)) filesystem::create_directory(OUTPUT_PATH);
	if (!filesystem::exists(BREAD_PATH)) filesystem::create_directory(BREAD_PATH);
	for (int i = 1; i <= NUMBER_OF_TRAYS; i++)
	{	// For each tray

		if (!filesystem::exists(PLATES_PATH + "tray" + to_string(i) + "/")) filesystem::create_directory(PLATES_PATH + "tray" + to_string(i) + "/");
		if (!filesystem::exists(OUTPUT_PATH + "tray" + to_string(i) + "/")) filesystem::create_directory(OUTPUT_PATH + "tray" + to_string(i) + "/");
		if (!filesystem::exists(BREAD_PATH + "tray" + to_string(i) + "/")) filesystem::create_directory(BREAD_PATH + "tray" + to_string(i) + "/");
		queue<BoundingBoxes> bb;
		for (const auto& imgname : IMAGE_NAMES)
		{	// For each image
			if (!filesystem::exists(PLATES_PATH + "tray" + to_string(i) + "/" + imgname + "/")) filesystem::create_directory(PLATES_PATH + "tray" + to_string(i) + "/" + imgname + "/");
			if (!filesystem::exists(BREAD_PATH + "tray" + to_string(i) + "/" + imgname + "/")) filesystem::create_directory(BREAD_PATH + "tray" + to_string(i) + "/" + imgname + "/");
			cv::Mat image = cv::imread(DATASET_PATH + "tray" + to_string(i) + "/" + imgname + ".jpg");
			bb.push(BoundingBoxes(image));
			vector<cv::Vec3f> plates = bb.back().getPlates();
			std::vector<cv::Rect> bread = bb.back().getBread();

			// Save plates cutouts
			for (int j = 0; j < plates.size(); j++)	cv::imwrite(PLATES_PATH + "tray" + to_string(i) + "/" + imgname + "/plate" + to_string(j) + ".jpg", cutout(image, plates[j]));
			// Save bread cutouts
			for (int j = 0; j < bread.size(); j++)	cv::imwrite(BREAD_PATH + "tray" + to_string(i) + "/" + imgname + "/bread" + to_string(j) + ".jpg", image(bread[j]));
		}

		// Plates segmentation
		if (!SKIP)
		{
			// Python OpenAI CLIP classifier
			if (DEBUG) cout << "Running Python script..." << endl;
			PyObject* pValue = PyLong_FromLong(i);
			PyTuple_SetItem(pArgs, 0, pValue);
			PyObject_CallObject(pFunc, pArgs);
			if (DEBUG) cout << "Python script finished" << endl;
		}

		// Segmentation
		for (const auto& imgname : IMAGE_NAMES)
		{	// For each image get the bounding boxes
			cv::Mat image = cv::imread(DATASET_PATH + "tray" + to_string(i) + "/" + imgname + ".jpg");
			vector<cv::Vec3f> plates = bb.front().getPlates();
			pair<bool, cv::Vec3f> salad = bb.front().getSalad();
			std::vector<cv::Rect> bread = bb.front().getBread();
			bb.pop();
			vector<string> files;
			cv::glob(PLATES_PATH + "tray" + to_string(i) + "/" + imgname + "/*.jpg", files);

			// tray_mask
			cv::Mat tray_mask = cv::Mat::zeros(image.size(), CV_8UC1);
			vector<string> boxes;

			//for metrics
			std::vector<std::pair<int, cv::Rect>> boxes_for_metrics;

			// Plates
			for (int j = 0; j < files.size(); j++)
			{	// For each plate in the image get the labels : association(file=plate_image, labels=categories, plates[j])
				vector<int> labels;
				ifstream infile(LABELS_PATH + files[j].substr(PLATES_PATH.length(), files[j].length() - 1) + ".txt");
				string category;
				while (getline(infile, category)) labels.push_back(stoi(category));
				infile.close();

				// Segmentation
				cv::Mat plate_image = cv::imread(files[j]);
				Segmentation seg(plate_image, labels);
				cv::Mat mask = seg.getSegments();
				vector<pair<int, cv::Rect>> box = seg.getBoxes();
				for (int k = 0; k < box.size(); k++)
				{
					int label = box[k].first;
					int x = box[k].second.x + plates[j][0] - plates[j][2];
					int y = box[k].second.y + plates[j][1] - plates[j][2];
					int w = box[k].second.width;
					int h = box[k].second.height;
					boxes.push_back("ID: " + to_string(label) + "; [" + to_string(x) + ", " + to_string(y) + ", " + to_string(w) + ", " + to_string(h) + "]");
					boxes_for_metrics.push_back(std::make_pair(label, cv::Rect(x, y, w, h)));
				}
				// add mask to tray_mask
				for (int k = 0; k < mask.rows; k++)
					for (int l = 0; l < mask.cols; l++)
						if (pow(k - plates[j][2], 2) + pow(l - plates[j][2], 2) <= pow(plates[j][2], 2)) //if (mask.at<uchar>(k,l) != 0)
							if (k + plates[j][1] - plates[j][2] >= 0 && k + plates[j][1] - plates[j][2] < tray_mask.rows && l + plates[j][0] - plates[j][2] >= 0 && l + plates[j][0] - plates[j][2] < tray_mask.cols)
								tray_mask.at<uchar>(k + plates[j][1] - plates[j][2], l + plates[j][0] - plates[j][2]) = mask.at<uchar>(k,l);
						
			}

			if (DEBUG) { cv::imshow("tray_mask", tray_mask * 15); cv::waitKey(0); }

			// Salad
			if (salad.first)
			{	// salad segmentation

				const int label = 12;
				cv::Mat salad_image = cutout(image, salad.second);

				//gamma correction
				cv::Mat gamma;
				cv::Mat lookUpTable(1, 256, CV_8U);
				uchar* p = lookUpTable.ptr();
				double gamma_ = 0.5;
				for (int i = 0; i < 256; ++i)
					p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, gamma_) * 255.0);
				cv::LUT(salad_image, lookUpTable, gamma);

				//to hsv
				cv::Mat hsv;
				cv::cvtColor(gamma, hsv, cv::COLOR_BGR2HSV);
				vector<cv::Mat> hsv_channels;
				cv::split(hsv, hsv_channels);
				cv::equalizeHist(hsv_channels[1], hsv_channels[1]);
				
				int sat = 206;

				cv::Mat mask;
				cv::threshold(hsv_channels[1], mask, sat, label, cv::THRESH_BINARY);
				mask = process(mask);

				//find bounding box of mask
				std::vector<std::vector<cv::Point>> contours;
				cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
				std::vector<cv::Rect> box(contours.size());
				for (size_t i = 0; i < contours.size(); i++)
					box[i] = cv::boundingRect(contours[i]);
				auto min = std::min_element(box.begin(), box.end(), [](const cv::Rect& a, const cv::Rect& b) {return a.area() < b.area(); });

				int x = min->x + salad.second[0] - salad.second[2];
				int y = min->y + salad.second[1] - salad.second[2];
				int w = min->width;
				int h = min->height;
				boxes.push_back("ID: " + to_string(label) + "; [" + to_string(x) + ", " + to_string(y) + ", " + to_string(w) + ", " + to_string(h) + "]");


				for (int k = 0; k < mask.rows; k++)
					for (int l = 0; l < mask.cols; l++)
						if (pow(k - salad.second[2],2) + pow(l - salad.second[2],2) <= pow(salad.second[2],2))
							if (k + salad.second[1] - salad.second[2] >= 0 && k + salad.second[1] - salad.second[2] < tray_mask.rows && l + salad.second[0] - salad.second[2] >= 0 && l + salad.second[0] - salad.second[2] < tray_mask.cols)
								tray_mask.at<uchar>(k + salad.second[1] - salad.second[2], l + salad.second[0] - salad.second[2]) = mask.at<uchar>(k, l);

				if (DEBUG) {
					cv::imshow("w/salad", tray_mask * 15);
					cv::waitKey(0);
				}
			}

			// Bread detection qith CLIP
			if (!SKIP)
			{
				// Python OpenAI CLIP classifier
				if (DEBUG) cout << "Running Python script..." << endl;
				PyObject* pValue = PyLong_FromLong(i);
				PyTuple_SetItem(pArgs, 0, pValue);
				PyObject_CallObject(pFunc_bread, pArgs);
				if (DEBUG) cout << "Python script finished" << endl;
			}

			// Bread
			if (!filesystem::is_empty(BREAD_OUT_PATH + "tray" + to_string(i) + "/" + imgname + "/"))
			{	// TODO: implement bread segmentation

				// get files in directory
				vector<string> files;
				for (const auto& entry : filesystem::directory_iterator(BREAD_OUT_PATH + "tray" + to_string(i) + "/" + imgname + "/"))
					files.push_back(entry.path().string());
				string breadimg = files[0].substr(0, files[0].size() - 4);	//remove .txt
				breadimg.replace(0, BREAD_OUT_PATH.size(), BREAD_PATH);		//replace path

				// read image
				cv::Mat bread = cv::imread(breadimg);
				cv::imshow("bread", bread);
				cv::waitKey(0);

				// TESTS DOWN HERE
				//gamma correction

				cv::Mat gamma;
				cv::Mat lookUpTable(1, 256, CV_8U);
				uchar* p = lookUpTable.ptr();
				double gamma_ = 0.5;
				for (int i = 0; i < 256; ++i)
					p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, gamma_) * 255.0);
				cv::LUT(bread, lookUpTable, gamma);

				//cv::circle(gamma, cv::Point(salad.second[0], salad.second[1]), salad.second[2], cv::Scalar(0, 0, 0), -1);
				//for (const auto circle : plates)
				//	cv::circle(gamma, cv::Point(circle[0], circle[1]), circle[2], cv::Scalar(0, 0, 0), -1);

				//to hsv
				cv::Mat hsv;
				cv::cvtColor(gamma, hsv, cv::COLOR_BGR2HSV);
				vector<cv::Mat> hsv_channels;
				cv::split(hsv, hsv_channels);
				cv::equalizeHist(hsv_channels[1], hsv_channels[1]);
				cv::merge(hsv_channels, hsv);
				//to rgb
				cv::Mat rgb;
				cv::cvtColor(hsv, rgb, cv::COLOR_HSV2BGR);

				int sat = 182;
				int bMin = 0;
				int bMax = 255;
				int gMin = 198;
				int gMax = 253;
				int rMin = 130;
				int rMax = 220;

				cv::namedWindow("trackbars", cv::WINDOW_NORMAL);
				cv::createTrackbar("sat", "trackbars", &sat, 255);
				cv::createTrackbar("bMin", "trackbars", &bMin, 255);
				cv::createTrackbar("bMax", "trackbars", &bMax, 255);
				cv::createTrackbar("gMin", "trackbars", &gMin, 255);
				cv::createTrackbar("gMax", "trackbars", &gMax, 255);
				cv::createTrackbar("rMin", "trackbars", &rMin, 255);
				cv::createTrackbar("rMax", "trackbars", &rMax, 255);


				while (DEBUG)
				{
					cv::Mat satr, ranged, original_sat, original_ran, mask;
					//cv::threshold(hsv_channels[1], satr, sat, 13, cv::THRESH_BINARY);
					//satr = process(satr);
					//cv::copyTo(gamma, original_sat, satr);
					//cv::inRange(original_sat, cv::Scalar(bMin, gMin, rMin), cv::Scalar(bMax, gMax, rMax), ranged);
					//mask = process(ranged);
					//cv::copyTo(gamma, original_ran, mask);
					//cv::imshow("ranged", original_ran);
					cv::inRange(rgb, cv::Scalar(bMin, gMin, rMin), cv::Scalar(bMax, gMax, rMax), mask);
					cv::imshow("filtered", mask);

					cv::Mat original;
					cv::bitwise_and(rgb, rgb, original, mask);

					cv::imshow("original", original);
					
					if (cv::waitKey(1) == 27) break;
					if (cv::waitKey(1) == 13)
					{
						cout << "Blue range: " << bMin << " " << bMax << endl;
						cout << "Green range: " << gMin << " " << gMax << endl;
						cout << "Red range: " << rMin << " " << rMax << endl;
						cv::Mat temp;
						cv::inRange(rgb, cv::Scalar(bMin, gMin, rMin), cv::Scalar(bMax, gMax, rMax), temp);
						cv::imshow("img", process(temp));
						cv::waitKey(0);
					}

					sat = cv::getTrackbarPos("sat", "trackbars");
					bMin = cv::getTrackbarPos("bMin", "trackbars");
					bMax = cv::getTrackbarPos("bMax", "trackbars");
					gMin = cv::getTrackbarPos("gMin", "trackbars");
					gMax = cv::getTrackbarPos("gMax", "trackbars");
					rMin = cv::getTrackbarPos("rMin", "trackbars");
					rMax = cv::getTrackbarPos("rMax", "trackbars");
				}
				cv::destroyAllWindows();

				cv::Mat mask;
				cv::threshold(hsv_channels[1], mask, sat, 13, cv::THRESH_BINARY);
				mask = process(mask);
			}

			//write bounding boxes
			if (!filesystem::exists(OUTPUT_PATH + "tray" + to_string(i) + "/bounding_boxes/")) filesystem::create_directory(OUTPUT_PATH + "tray" + to_string(i) + "/bounding_boxes/");
			for (int k = 0; k < boxes.size(); k++)
			{
				if (DEBUG) cout << imgname + " " + boxes[k] << endl;

				std::ofstream file;
				string path = OUTPUT_PATH + "tray" + to_string(i) + "/bounding_boxes/" + imgname + "_bounding_boxes.txt";

				if (k == 0)
					file = std::ofstream(path);
				else
					file = std::ofstream(path, std::ios_base::app);


				if (file.is_open())
				{
					if (k < boxes.size() - 1)
						file << boxes[k] << endl;
					else
						file << boxes[k];
				}
			}

			//write tray mask
			if (!filesystem::exists(OUTPUT_PATH + "tray" + to_string(i) + "/masks/")) filesystem::create_directory(OUTPUT_PATH + "tray" + to_string(i) + "/masks/");
			cv::imwrite(OUTPUT_PATH + "tray" + to_string(i) + "/masks/" + imgname + "_mask.png", tray_mask);


			//Metrics
			
			//read original files
			const string BOXES_PATH = DATASET_PATH + "tray" + to_string(i) + "/bounding_boxes/";
			const string MASK_PATH = DATASET_PATH + "tray" + to_string(i) + "/masks/";
			vector<string> boxes_files, mask_files;
			cv::glob(BOXES_PATH + "*.txt", boxes_files);
			cv::glob(MASK_PATH + "*", mask_files);
			
			for (int k = 0; k < boxes_files.size(); k++)
			{
				//read boxes
				vector<pair<int, cv::Rect>> original_boxes;
				std::ifstream file(boxes_files[k]);
				if (file.is_open())
				{
					string line;
					while (getline(file, line))
					{
						size_t id_start = line.find(":") + 2;
						size_t id_end = line.find(";");
						size_t box_start = line.find("[") + 1;
						size_t box_end = line.find("]");
						string id = line.substr(id_start, id_end - id_start);
						string box = line.substr(box_start, box_end - box_start);
						//split box values
						istringstream iss(box);
						vector<string> tokens{ istream_iterator<string>{iss}, istream_iterator<string>{} };
						cv::Rect tmp(stoi(tokens[0]), stoi(tokens[1]), stoi(tokens[2]), stoi(tokens[3]));
						original_boxes.push_back(make_pair(stoi(id), tmp));
					}
					file.close();
				}

				cv::Mat original_mask = cv::imread(mask_files[k]);
				Metrics met(tray_mask, boxes_for_metrics, original_mask, original_boxes);
				
			}

		}
	}

	// Python finalization
	Py_Finalize();

	return 0;
}
