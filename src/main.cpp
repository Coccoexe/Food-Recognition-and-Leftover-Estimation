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

#include <Python.h>

#define DEBUG false

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
	}

	return 0;
}
