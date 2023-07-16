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
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/objdetect.hpp>

#include <opencv2/ximgproc.hpp>
#include <opencv2/intensity_transform.hpp>

using namespace std;

vector<cv::Vec3b> colors_mouse;

void onMouse(int event, int x, int y, int f, void* userdata)
{
	if (event == cv::EVENT_LBUTTONDOWN)
	{
		//get image
		cv::Mat img = *(cv::Mat*)userdata;

		//get pixel color
		cv::Vec3b color = img.at<cv::Vec3b>(y, x);
		colors_mouse.push_back(color);

		//print pixel color
		cout << "Pixel BGR value: " << color << endl;
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

int main111()
{
	cv::Mat img = cv::imread("./Food_leftover_dataset/tray1/food_image.jpg");
	cv::Mat mask = cv::imread("./Food_leftover_dataset/tray1/masks/food_image_mask.png");

	for (int y = 0; y < mask.rows; y++)
		for (int x = 0; x < mask.cols; x++)
			mask.at<uint8_t>(y, x) *= 10;
	cv::imshow("mask", mask);
	cv::waitKey(0);


	//gamma transform
	cv::Mat gamma;
	cv::Mat lookUpTable(1, 256, CV_8U);
	uchar* p = lookUpTable.ptr();
	double gamma_ = 0.5;
	for (int i = 0; i < 256; ++i)
		p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, gamma_) * 255.0);
	cv::LUT(img, lookUpTable, gamma);
	cv::imshow("gamma", gamma);
	cv::waitKey(0);

	//image to hsv
	cv::Mat hsv;
	cv::cvtColor(gamma, hsv, cv::COLOR_BGR2HSV);

	//enhance saturation
	cv::Mat hsv_enhanced;
	vector<cv::Mat> hsv_channels;
	cv::split(hsv, hsv_channels);
	cv::equalizeHist(hsv_channels[1], hsv_channels[1]);
	//hsv_channels[1] *= double(saturation)/10.0;
	cv::merge(hsv_channels, hsv_enhanced);
	cv::cvtColor(hsv_enhanced, hsv_enhanced, cv::COLOR_HSV2BGR);

	for (int i = 1; i < 14; i++)
	{
		vector<int> b, g, r;
		//if in mask there is a pixel with value i, add its repcective color to the vector
		for (int y = 0; y < mask.rows; y++)
			for (int x = 0; x < mask.cols; x++)
				if (mask.at<uchar>(y, x) == i)
				{
					b.push_back(hsv_enhanced.at<cv::Vec3b>(y, x)[0]);
					g.push_back(hsv_enhanced.at<cv::Vec3b>(y, x)[1]);
					r.push_back(hsv_enhanced.at<cv::Vec3b>(y, x)[2]);
				}
		//calculate the mean color
		int sum = 0, mean_b = 0, mean_g = 0, mean_r = 0;
		if (b.size() > 0)
		{
			for (int j = 0; j < b.size(); j++)
				sum += b[j];
			mean_b = sum / b.size();
		}
		sum = 0;
		if (g.size() > 0)
		{
			for (int j = 0; j < g.size(); j++)
				sum += g[j];
			mean_g = sum / g.size();
		}
		sum = 0;
		if (r.size() > 0)
		{
			for (int j = 0; j < r.size(); j++)
				sum += r[j];
			mean_r = sum / r.size();
		}

		//print the mean color
		std::cout << "Mean color for class " << i << ": " << mean_b << " " << mean_g << " " << mean_r << endl;


	}
	cv::imshow("img", img);
	cv::imshow("mask", mask);
	cv::waitKey(0);
	return 0;
}

int main()
{
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

	const bool DEBUG = true;
	const bool TEST = !true;

	string output = "C:\\Users\\alessio\\Desktop\\output\\";
	string input = "./Food_leftover_dataset/";
	cv::Mat img;
	string name, outname;

	if (!TEST)
	{
		img = cv::imread(input + "tray1/food_image.jpg");
		//img = cv::imread("C:\\Users\\alessio\\Desktop\\3.png");

		//gamma transform
		cv::Mat gamma;
		cv::Mat lookUpTable(1, 256, CV_8U);
		uchar* p = lookUpTable.ptr();
		double gamma_ = 0.5;
		for (int i = 0; i < 256; ++i)
			p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, gamma_) * 255.0);
		cv::LUT(img, lookUpTable, gamma);

		//image to hsv
		cv::Mat hsv;
		cv::cvtColor(gamma, hsv, cv::COLOR_BGR2HSV);

		//enhance saturation
		cv::Mat hsv_enhanced;
		vector<cv::Mat> hsv_channels;
		cv::split(hsv, hsv_channels);
		cv::equalizeHist(hsv_channels[1], hsv_channels[1]);
		//hsv_channels[1] *= 5;
		cv::merge(hsv_channels, hsv_enhanced);
		cv::cvtColor(hsv_enhanced, hsv_enhanced, cv::COLOR_HSV2BGR);

		//bounding boxes
		vector<pair<int, cv::Rect>> bounding_boxes;
		bounding_boxes.push_back(make_pair(1, cv::Rect(737, 145, 384, 400)));

		//set to 0 outside the bounding boxes
		for (int y = 0; y < hsv_enhanced.rows; y++)
			for (int x = 0; x < hsv_enhanced.cols; x++)
			{
				bool found = false;
				for (int i = 0; i < bounding_boxes.size(); i++)
					if (bounding_boxes[i].second.contains(cv::Point(x, y)))
					{
						found = true;
						break;
					}
				if (!found)
					hsv_enhanced.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
			}

		cv::Mat msk1, out;
		cv::inRange(hsv_enhanced, cv::Scalar(0, 105, 150), cv::Scalar(16, 168, 255), msk1); //PASTA PESTO
		out = process(msk1, img);
		cv::imshow("out", out);
		msk1 = out.clone();

		//hsv to gray
		cv::Mat gray = cv::Mat::zeros(hsv_enhanced.size(), CV_8UC1);
		gray(bounding_boxes[0].second).setTo(cv::GC_PR_BGD);
		cv::threshold(out, out, 0, 1, cv::THRESH_BINARY);
		cv::Mat somma;
		cv::addWeighted(out, 1, gray, 1, 0, somma);
		cv::imshow("somma", somma * 100);

		cv::Mat segments = cv::Mat::zeros(img.size(), CV_8UC1);
		cv::Mat bgdModel, fgdModel;
		cv::grabCut(img, somma, cv::Rect(), bgdModel, fgdModel, 10, cv::GC_INIT_WITH_MASK);
		cv::Mat1b mask_fgpf = (somma == cv::GC_FGD) | (somma == cv::GC_PR_FGD);
		cv::Mat3b tmp = cv::Mat3b::zeros(img.rows, img.cols);
		img.copyTo(tmp, mask_fgpf);
		cv::erode(tmp, tmp, cv::Mat());
		cv::imshow("foreground", tmp);
		cv::waitKey(0);
		cv::cvtColor(msk1, msk1, cv::COLOR_GRAY2BGR);
		cv::addWeighted(tmp, 1, msk1, 0.5, 0, tmp);
		cv::imshow("foreground", tmp);
		cv::waitKey(0);

		int bMin = 0;
		int bMax = 255;
		int gMin = 0;
		int gMax = 255;
		int rMin = 0;
		int rMax = 255;
		cv::Mat filtered;

		//trackbar for rbg values
		cv::namedWindow("trackbar");
		cv::createTrackbar("bMin", "trackbar", 0, 255);
		cv::createTrackbar("bMax", "trackbar", 0, 255);
		cv::createTrackbar("gMin", "trackbar", 0, 255);
		cv::createTrackbar("gMax", "trackbar", 0, 255);
		cv::createTrackbar("rMin", "trackbar", 0, 255);
		cv::createTrackbar("rMax", "trackbar", 0, 255);

		//loop for trackbar
		while (DEBUG)
		{
			cv::inRange(hsv_enhanced, cv::Scalar(bMin, gMin, rMin), cv::Scalar(bMax, gMax, rMax), filtered);
			cv::imshow("filtered", filtered);

			cv::Mat original;
			cv::bitwise_and(hsv_enhanced, hsv_enhanced, original, filtered);

			cv::imshow("original", original);

			if (cv::waitKey(1) == 27)
				break;

			//if enter is pressed, save the values
			if (cv::waitKey(1) == 13)
			{
				cout << "Blue range: " << bMin << " " << bMax << endl;
				cout << "Green range: " << gMin << " " << gMax << endl;
				cout << "Red range: " << rMin << " " << rMax << endl;
				cv::Mat temp;
				cv::inRange(hsv_enhanced, cv::Scalar(bMin, gMin, rMin), cv::Scalar(bMax, gMax, rMax), temp);
				cv::imshow("img", process(temp, img));
				cv::waitKey(0);
			}

			rMin = cv::getTrackbarPos("rMin", "trackbar");
			rMax = cv::getTrackbarPos("rMax", "trackbar");
			gMin = cv::getTrackbarPos("gMin", "trackbar");
			gMax = cv::getTrackbarPos("gMax", "trackbar");
			bMin = cv::getTrackbarPos("bMin", "trackbar");
			bMax = cv::getTrackbarPos("bMax", "trackbar");
		}

		cv::destroyAllWindows();
	}

	if (TEST)
	{
		for (int k = 1; k < 9; k++)
		{
			for (int l = 0; l < 4; l++)
			{
				switch (l)
				{
				case 0:
					name = ("tray" + to_string(k) + "\\food_image.jpg");
					outname = to_string(k) + "_foodImage";
					break;
				case 1:
					name = ("tray" + to_string(k) + "\\leftover1.jpg");
					outname = to_string(k) + "_leftover1";
					break;
				case 2:
					name = ("tray" + to_string(k) + "\\leftover2.jpg");
					outname = to_string(k) + "_leftover2";
					break;
				case 3:
					name = ("tray" + to_string(k) + "\\leftover3.jpg");
					outname = to_string(k) + "_leftover3";
					break;
				}

				cout << "Processing" << name << endl;

				img = cv::imread(input + name);

				//gamma transform
				cv::Mat gamma;
				cv::Mat lookUpTable(1, 256, CV_8U);
				uchar* p = lookUpTable.ptr();
				double gamma_ = 0.5;
				for (int i = 0; i < 256; ++i)
					p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, gamma_) * 255.0);
				cv::LUT(img, lookUpTable, gamma);
				//cv::imshow("gamma", gamma);
				//cv::waitKey(0);

				//image to hsv
				cv::Mat hsv;
				cv::cvtColor(gamma, hsv, cv::COLOR_BGR2HSV);

				//enhance saturation
				cv::Mat hsv_enhanced;
				vector<cv::Mat> hsv_channels;
				cv::split(hsv, hsv_channels);
				//cv::equalizeHist(hsv_channels[0], hsv_channels[0]);
				cv::equalizeHist(hsv_channels[1], hsv_channels[1]);
				//cv::equalizeHist(hsv_channels[2], hsv_channels[2]);
				//hsv_channels[1] *= double(saturation)/10.0;
				cv::merge(hsv_channels, hsv_enhanced);
				cv::cvtColor(hsv_enhanced, hsv_enhanced, cv::COLOR_HSV2BGR);

				cv::Mat msk1, out;

				//cv::inRange(hsv_enhanced, cv::Scalar(0, 105, 150), cv::Scalar(16, 168, 255), msk1); //PASTA PESTO
				//out = process(msk1, img);
				//cv::imwrite(output + outname + "_pesto.jpg", out);

				//cv::inRange(hsv_enhanced, cv::Scalar(0, 139, 153), cv::Scalar(52, 255, 255), msk1); //PASTA POMODORO
				//out = process(msk1, img);
				//cv::imwrite(output + outname + "_pomodoro.jpg", out);

				//cv::inRange(hsv_enhanced, cv::Scalar(0, 127, 100), cv::Scalar(24, 246, 238), msk1); //PASTA RAGU
				//out = process(msk1, img);
				//cv::imwrite(output + outname + "_ragu.jpg", out);

				cv::inRange(hsv_enhanced, cv::Scalar(9, 23, 156), cv::Scalar(56, 64, 255), msk1); //FAGIOLI
				out = process(msk1, img);
				cv::imwrite(output + outname + "_fagioli.jpg", out);

				//cv::inRange(hsv_enhanced, cv::Scalar(50, 95, 158), cv::Scalar(87, 194, 217), msk1); //CARNE
				//out = process(msk1, img);
				//cv::imwrite(output + outname + "_carne.jpg", out);


				//cv::inRange(hsv_enhanced, cv::Scalar(0, 45, 140), cv::Scalar(35, 100, 165), msk1); //CONIGLIO
				//out = process(msk1, img);
				//cv::imwrite(output + outname + "_coniglio.jpg", out);

				//cv::inRange(hsv_enhanced, cv::Scalar(0, 134, 205), cv::Scalar(29, 175, 241), msk1); //PASTA COZZE
				//out = process(msk1, img);
				//cv::imwrite(output + outname + "_cozze.jpg", out);

				//cv::inRange(hsv_enhanced, cv::Scalar(9, 90, 150), cv::Scalar(40, 135, 205), msk1); //PESCE
				//out = process(msk1, img);
				//cv::imwrite(output + outname + "_pesce.jpg", out);

				//cv::inRange(hsv_enhanced, cv::Scalar(56, 150, 210), cv::Scalar(100, 190, 245), msk1); //RISO
				//out = process(msk1, img);
				//cv::imwrite(output + outname + "_riso.jpg", out);

				//cv::inRange(hsv_enhanced, cv::Scalar(68, 190, 220), cv::Scalar(113, 255, 255), msk1); //PATATE
				//out = process(msk1, img);
				//cv::imwrite(output + outname + "_patate.jpg", out);
			}
		}
		return 0;
	}

	return 0;
}

int mainzzz()
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

	Mask m_10(cv::imread(dataset + "tray1/food_image.jpg"));                                      // Mask object for tray 1
	Mask m_11(cv::imread(dataset + "tray1/leftover1.jpg"));                                       // Mask object for tray 1
	Mask m_12(cv::imread(dataset + "tray1/leftover2.jpg"));                                       // Mask object for tray 1
	Mask m_13(cv::imread(dataset + "tray1/leftover3.jpg"));                                       // Mask object for tray 1
	Mask m_20(cv::imread(dataset + "tray2/food_image.jpg"));                                      // Mask object for tray 2
	Mask m_21(cv::imread(dataset + "tray2/leftover1.jpg"));                                       // Mask object for tray 2
	Mask m_22(cv::imread(dataset + "tray2/leftover2.jpg"));                                       // Mask object for tray 2
	Mask m_23(cv::imread(dataset + "tray2/leftover3.jpg"));                                       // Mask object for tray 2
	Mask m_30(cv::imread(dataset + "tray3/food_image.jpg"));                                      // Mask object for tray 3
	Mask m_31(cv::imread(dataset + "tray3/leftover1.jpg"));                                       // Mask object for tray 3
	Mask m_32(cv::imread(dataset + "tray3/leftover2.jpg"));                                       // Mask object for tray 3
	Mask m_33(cv::imread(dataset + "tray3/leftover3.jpg"));                                       // Mask object for tray 3
	Mask m_40(cv::imread(dataset + "tray4/food_image.jpg"));                                      // Mask object for tray 4
	Mask m_41(cv::imread(dataset + "tray4/leftover1.jpg"));                                       // Mask object for tray 4
	Mask m_42(cv::imread(dataset + "tray4/leftover2.jpg"));                                       // Mask object for tray 4
	Mask m_43(cv::imread(dataset + "tray4/leftover3.jpg"));                                       // Mask object for tray 4
	Mask m_50(cv::imread(dataset + "tray5/food_image.jpg"));                                      // Mask object for tray 5
	Mask m_51(cv::imread(dataset + "tray5/leftover1.jpg"));                                       // Mask object for tray 5
	Mask m_52(cv::imread(dataset + "tray5/leftover2.jpg"));                                       // Mask object for tray 5
	Mask m_53(cv::imread(dataset + "tray5/leftover3.jpg"));                                       // Mask object for tray 5
	Mask m_60(cv::imread(dataset + "tray6/food_image.jpg"));                                      // Mask object for tray 6
	Mask m_61(cv::imread(dataset + "tray6/leftover1.jpg"));                                       // Mask object for tray 6
	Mask m_62(cv::imread(dataset + "tray6/leftover2.jpg"));                                       // Mask object for tray 6
	Mask m_63(cv::imread(dataset + "tray6/leftover3.jpg"));                                       // Mask object for tray 6
	Mask m_70(cv::imread(dataset + "tray7/food_image.jpg"));                                      // Mask object for tray 7
	Mask m_71(cv::imread(dataset + "tray7/leftover1.jpg"));                                       // Mask object for tray 7
	Mask m_72(cv::imread(dataset + "tray7/leftover2.jpg"));                                       // Mask object for tray 7
	Mask m_73(cv::imread(dataset + "tray7/leftover3.jpg"));                                       // Mask object for tray 7
	Mask m_80(cv::imread(dataset + "tray8/food_image.jpg"));                                      // Mask object for tray 8
	Mask m_81(cv::imread(dataset + "tray8/leftover1.jpg"));                                       // Mask object for tray 8
	Mask m_82(cv::imread(dataset + "tray8/leftover2.jpg"));                                       // Mask object for tray 8
	Mask m_83(cv::imread(dataset + "tray8/leftover3.jpg"));                                       // Mask object for tray 8





	int tray = 1;                                                                                 // Tray number
	cv::Mat image = cv::imread(dataset + "tray" + to_string(tray) + "/food_image.jpg");
	vector<pair<int, cv::Rect>> box;
	switch (tray)
	{
	case 1:
		box.push_back(pair<int, cv::Rect>(6, cv::Rect(370, 436, 313, 331)));
		box.push_back(pair<int, cv::Rect>(1, cv::Rect(737, 145, 384, 400)));
		box.push_back(pair<int, cv::Rect>(10, cv::Rect(259, 532, 347, 357)));
		box.push_back(pair<int, cv::Rect>(13, cv::Rect(235, 79, 243, 178)));
		break;
	case 2:
		box.push_back(pair<int, cv::Rect>(2, cv::Rect(747, 450, 365, 410)));
		box.push_back(pair<int, cv::Rect>(7, cv::Rect(563, 130, 144, 238)));
		box.push_back(pair<int, cv::Rect>(12, cv::Rect(150, 451, 355, 346)));
		box.push_back(pair<int, cv::Rect>(11, cv::Rect(368, 122, 262, 278)));
		break;
	case 3:
		box.push_back(pair<int, cv::Rect>(12, cv::Rect(163, 481, 352, 337)));
		box.push_back(pair<int, cv::Rect>(2, cv::Rect(744, 452, 352, 397)));
		box.push_back(pair<int, cv::Rect>(8, cv::Rect(371, 140, 292, 305)));
		break;
	case 4:
		box.push_back(pair<int, cv::Rect>(12, cv::Rect(805, 56, 417, 352)));
		box.push_back(pair<int, cv::Rect>(5, cv::Rect(574, 508, 444, 376)));
		box.push_back(pair<int, cv::Rect>(11, cv::Rect(188, 127, 357, 340)));
		box.push_back(pair<int, cv::Rect>(13, cv::Rect(109, 565, 227, 277)));
		box.push_back(pair<int, cv::Rect>(7, cv::Rect(384, 117, 219, 266)));
		break;
	case 5:
		box.push_back(pair<int, cv::Rect>(10, cv::Rect(730, 519, 378, 284)));
		box.push_back(pair<int, cv::Rect>(13, cv::Rect(516, 0, 214, 222)));
		box.push_back(pair<int, cv::Rect>(8, cv::Rect(731, 294, 293, 389)));
		box.push_back(pair<int, cv::Rect>(3, cv::Rect(163, 305, 403, 393)));
		break;
	case 6:
		box.push_back(pair<int, cv::Rect>(6, cv::Rect(762, 547, 276, 334)));
		box.push_back(pair<int, cv::Rect>(12, cv::Rect(845, 103, 305, 350)));
		box.push_back(pair<int, cv::Rect>(10, cv::Rect(599, 505, 260, 379)));
		box.push_back(pair<int, cv::Rect>(4, cv::Rect(225, 139, 383, 354)));
		break;
	case 7:
		box.push_back(pair<int, cv::Rect>(7, cv::Rect(612, 664, 256, 191)));
		box.push_back(pair<int, cv::Rect>(12, cv::Rect(824, 112, 306, 333)));
		box.push_back(pair<int, cv::Rect>(11, cv::Rect(641, 521, 318, 285)));
		box.push_back(pair<int, cv::Rect>(4, cv::Rect(223, 135, 385, 350)));
		break;
	case 8:
		box.push_back(pair<int, cv::Rect>(9, cv::Rect(567, 450, 276, 283)));
		box.push_back(pair<int, cv::Rect>(12, cv::Rect(852, 125, 310, 296)));
		box.push_back(pair<int, cv::Rect>(10, cv::Rect(660, 648, 274, 251)));
		box.push_back(pair<int, cv::Rect>(11, cv::Rect(801, 486, 174, 301)));
		box.push_back(pair<int, cv::Rect>(4, cv::Rect(247, 152, 367, 342)));
		break;
	default:
		return -1;
		break;
	}
	cv::imshow("image", image);
	//Mask m(image, box);
	Mask m(image);

	return 0;

	std::multimap<int, const cv::Mat> match;                                                      // Map of matching images
	for (int i = 1; i < labels.size(); i++)
	{   // Process each LABEL of images that will be used for matching
		string folder = matching + labels.at(i) + "/";                                            // Matching folder name

		for (const auto& entry : filesystem::directory_iterator(folder))
		{   // Process each IMAGE in the folder
			const cv::Mat img = cv::imread(entry.path().string());
			return 0;// Read image
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
			//mm.push_back(Mask(img, bb.at(image)));                                                // Create Mask object
		}
		tt.push_back(Metrics(bb, mm)); 															  // Create Metrics object
	}
	return 0;
}

// OLD CODE

/*
cv::Mat img1 = cv::imread(matching + "image.png");
	cv::Mat img2 = cv::imread(dataset + "tray1/food_image.jpg");

	//SURF

	cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(400);
	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	cv::Mat descriptors1, descriptors2;
	detector->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
	detector->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);

	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
	std::vector<std::vector<cv::DMatch>> knn_matches;
	matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);

	const float ratio_thresh = 0.7f;
	std::vector<cv::DMatch> good_matches;
	for (size_t i = 0; i < knn_matches.size(); i++)
	{
		if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
		{
			good_matches.push_back(knn_matches[i][0]);
		}
	}

	cv::Mat img_matches;
	cv::drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, cv::Scalar::all(-1),
		cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	cv::imshow("Good Matches", img_matches);
	cv::waitKey(0);
	return 0;
*/

/*
// grabcut segmentation (2 classes) inside bounding boxes
	for (int i = 0; i < box.size(); i++) {
		cv::Mat mask, bgdModel, fgdModel;
		cv::Rect rect(box.at(i)[1], box.at(i)[2], box.at(i)[3], box.at(i)[4]);
		cv::grabCut(sgm, mask, rect, bgdModel, fgdModel, 1, cv::GC_INIT_WITH_RECT);
		cv::compare(mask, cv::GC_PR_FGD, mask, cv::CMP_EQ);
		cv::Mat foreground(sgm.size(), CV_8UC3, cv::Scalar(255, 255, 255));
		sgm.copyTo(foreground, mask);
		cv::imshow("foreground", foreground);
		cv::waitKey(0);
	}
*/

/*
// grabcut the whole image in box.size() + 1 (background) classes, each class can't exceed the bounding box, then show the result (black background):
	cv::Mat segments = cv::Mat::zeros(sgm.size(), CV_8UC1);                                // Final mask
	for (int i = 0; i < box.size(); i++)
	{   // Process each BOUNDING BOX
		cv::Mat window = cv::Mat::zeros(sgm.size(), CV_8UC1);                              // Mask for the current bounding box
		cv::Rect rect(box.at(i).at(1), box.at(i).at(2), box.at(i).at(3), box.at(i).at(4)); // Bounding box
		cv::Mat bgModel, fgModel;                                                          // Background and foreground models
		cv::grabCut(sgm, window, rect, bgModel, fgModel, 3, cv::GC_INIT_WITH_RECT);        // GrabCut
		cv::compare(window, cv::GC_PR_FGD, window, cv::CMP_EQ);                            // Mask of probable foreground pixels
		window = window / 255;                                                             // Convert to 0-1 values
		cv::medianBlur(window, window, 5);                                                 // Median blur to remove noise and fill holes
		cv::add(segments, window * (box.at(i).at(0) * 255 / 13), segments);                // Add the current mask to the final mask with intensity proportional to the label
	}
	cv::imshow("mask", segments);
	cv::waitKey(0);
*/

/*
	// the image gets divided into regions of interest (ROIs) based on the bounding boxes:
	// each ROI is then segmented separately using k-means clustering, where k = N (concurrent bounding boxes) + 1 (background) classes
	// each ROI's mask is then added to the final mask

	cv::Mat overlap = cv::Mat::zeros(sgm.size(), CV_8UC1);                // Mask of overlapping regions
	for (int i = 0; i < box.size(); i++) overlap(box.at(i).second) += 19; // Add bounding box to overlap mask with intensity 1
	cv::imshow("overlap", overlap);
	cv::waitKey(0);
*/

/*
// k-means clustering of source_image, with k = bounding_boxes.size() + 1
	cv::Mat samples(source_image.rows * source_image.cols, 3, CV_32F);
	for (int y = 0; y < source_image.rows; y++)
		for (int x = 0; x < source_image.cols; x++)
			for (int z = 0; z < 3; z++)
				samples.at<float>(y + x * source_image.rows, z) = source_image.at<cv::Vec3b>(y, x)[z];

	int clusterCount = bounding_boxes.size() + 1;
	cv::Mat labels2;
	int attempts = 5;
	cv::Mat centers;
	cv::kmeans(samples, clusterCount, labels2, cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 10000, 0.0001), attempts, cv::KMEANS_PP_CENTERS, centers);
	cv::Mat new_image(source_image.size(), source_image.type());
	for (int y = 0; y < source_image.rows; y++)
		for (int x = 0; x < source_image.cols; x++)
		{
			int cluster_idx = labels2.at<int>(y + x * source_image.rows, 0);
			new_image.at<cv::Vec3b>(y, x)[0] = centers.at<float>(cluster_idx, 0);
			new_image.at<cv::Vec3b>(y, x)[1] = centers.at<float>(cluster_idx, 1);
			new_image.at<cv::Vec3b>(y, x)[2] = centers.at<float>(cluster_idx, 2);
		}
	cv::medianBlur(new_image, new_image, 3);
	cv::dilate(new_image, new_image, cv::Mat(), cv::Point(-1, -1), 2, 1, 1);
	cv::erode(new_image, new_image, cv::Mat(), cv::Point(-1, -1), 2, 1, 1);
	cv::imshow("clustered image", new_image);
	cv::waitKey(0);
*/