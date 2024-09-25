// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

//#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
//#include <opencv2/opencv.hpp>   // Include OpenCV API
//#include <iostream>
//using namespace std;
//using namespace cv;

//-------------------(BTN1)_câu c_bài 2---------------------------------
//#include <opencv2/opencv.hpp>   // Include OpenCV API
//#include <iostream>
//using namespace std;
//using namespace cv;
//int main() {
//	Mat source = (Mat_<uint8_t>(4,4)<<60,20,30,60,
//									80,60,100,110,
//									120,160,60,150,
//									220,230,240,250);
//	Mat dst;
//	double thresh = 0;
//	double maxValue = 255;
//	long double thres = cv::threshold(source, dst, thresh, maxValue, THRESH_OTSU);
//	cout << "INPUT" << endl << source << endl;
//	cout << endl;
//	cout << "Otsu Threshold T=" << thres << endl;
//	cout << endl;
//	cout << "OUTPUT" << endl << "" << dst << endl;
//	return 0;
//}


//------------------------(BTN1)_câu b_bài 2--------------------------
////#include <opencv2/opencv.hpp>   // Include OpenCV API
//#include <iostream>
//using namespace std;
//using namespace cv;
// 
//void histogram(string const& name, Mat const& Image)
//{
//	int bin = 256;
//	int histsize[] = { bin };
//	float range[] = { 0, 255 };
//	const float* ranges[] = { range };
//	Mat hist;
//	int channel[] = { 0 };
//	int hist_height = 255;
//	Mat hist_image = Mat::zeros(hist_height, bin, CV_8SC3);
//	calcHist(&Image, 1, channel, Mat(), hist, 1, histsize, ranges, true, false);
//	double max_val = 0;
//	minMaxLoc(hist, 0, &max_val);
//	for (int i = 0; i < bin; i++)
//	{
//		float binV = hist.at<float>(i);
//		int height = cvRound(binV * hist_height / max_val);
//		line(hist_image, Point(i, hist_height), Point(i, hist_height - height),
//			Scalar::all(255));
//	}
//	imshow(name, hist_image);
//}
//int main(int argc, char** argv)
//{
//	float anh[16] = { 60, 20, 30, 60, 80, 60, 100, 110, 120, 160, 60, 150, 220, 230, 240, 250 };
//	Mat gray_anh = Mat(4, 4, CV_32F, anh);
//	Mat gray_his;
//	namedWindow("Gray_old", WINDOW_FREERATIO);
//	gray_anh.convertTo(gray_anh, CV_8UC1);
//	equalizeHist(gray_anh, gray_his);
//	cout << "Matrix of the picture" << endl << gray_anh << endl << endl;
//	cout << "Picture after equalization" << endl << gray_his << endl << endl;
//	imshow("Gray_old", gray_anh);
//	imshow("Gray_new_histogram", gray_his);
//	histogram("Histogram before equalization", gray_anh);
//	histogram("Histogram after equalization", gray_his);
//	waitKey(0);
//	return 0;
//}


//----------------(BTN1)_câu a_bài 4-------------------
//#include <iostream>
//#include <opencv2/opencv.hpp>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui.hpp>
//
//
//int main()
//{
//    std::cout << "Hello World!\n";
//    float sigma = 0.6;
//    cv::Mat vector = (cv::Mat_<float>(5, 7) <<
//        9, 10, 11, 10, 9, 10, 11,
//        10, 9, 110, 110, 110, 10, 11,
//        10, 110, 10, 11, 10, 110, 11,
//        10, 9, 110, 110, 110, 13, 11,
//        10, 10, 10, 10, 10, 13, 11);
//    cv::Mat dst;
//    cv::GaussianBlur(vector, dst, cv::Size(3, 3), sigma, sigma, CV_HAL_BORDER_REPLICATE);
//    std::cout << "GaussianBlur\n" << dst << std::endl;
//    return 0;
//}
// 
// 
//------------------(BTN1)_câu b_bài 4---------------------
//#include <iostream>
//#include <opencv2/opencv.hpp>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui.hpp>
//using namespace std;
//using namespace cv;
//
//int main()
//{
//    cout << "Hello World!\n";
//    float sigma = 0.6;
//    Mat vector = (Mat_<float>(5, 7) <<
//        9, 10, 11, 10, 9, 10, 11,
//        10, 9, 110, 110, 110, 10, 11,
//        10, 110, 10, 11, 10, 110, 11,
//        10, 9, 110, 110, 110, 13, 11,
//        10, 10, 10, 10, 10, 13, 11);
//    Mat kernelx = (Mat_<double>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
//    Mat kernely = (Mat_<double>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
//    //Calculate sobel along x-axis
//    Mat sobelx;
//    Mat filter2d_x;
//    Sobel(vector, sobelx, CV_64F, 1, 0, 3, 1, 0, BORDER_REPLICATE);
//    filter2D(vector, filter2d_x, -1, kernelx, Point(-1, -1), 0, CV_HAL_BORDER_REPLICATE);
//    cout << "Sobel_x=" << endl << sobelx << endl;
//    //Calculate sobel along y_axis
//    Mat sobely;
//    Mat filter2d_y;
//    Sobel(vector, sobely, CV_64F, 0, 1, 3, 1, 0, BORDER_REPLICATE);
//    filter2D(vector, filter2d_y, -1, kernely, Point(-1, -1), 0, CV_HAL_BORDER_REPLICATE);
//    cout << endl;
//    cout << "Sobel_y=" << endl << sobely << endl;
//    return 0;
//}




//-------------------(BTN1)_câu a_bài 5------------------------
//#include <opencv2/opencv.hpp>
//#include <iostream>
//using namespace cv;
//int main() {
//	Mat img = imread("Picture4.png", IMREAD_GRAYSCALE);
//	// Check if the image loaded successfully
//	if (img.empty()) {
//		std::cerr << "Error: Could not load the image!" << std::endl;
//		return -1;
//	}
//	// Resize the image if necessary (adjust size as needed)
//	resize(img, img, Size(100, 100));
//	// Threshold to convert to binary (black and white)
//	Mat binary_img;
//	threshold(img, binary_img, 128, 255, THRESH_BINARY);
//	// Define radii and initialize the center of the image
//	int radii[] = { 1, 5, 10, 15, 20, 25, 30, 35, 40 };
//	Point center(img.cols / 2, img.rows / 2);
//	// Loop through each radius and count the white pixels
//	for (int radius : radii) {
//		// Create a mask for the circle
//		Mat mask = Mat::zeros(binary_img.size(), CV_8UC1);
//		circle(mask, center, radius, Scalar(255), -1);  // Draw filled circle
//		// Count the number of white pixels within the circle
//		Mat masked_img;
//		bitwise_and(binary_img, binary_img, masked_img, mask);
//		int white_pixels = countNonZero(masked_img);
//		// Output the radius and white pixel count
//		std::cout << "Radius: " << radius << " - White Pixels: " << white_pixels << std::endl;
//	}
//	return 0;
//}


//-------------------(BTN1)_câu b_bài 5------------------------
//#include <opencv2/opencv.hpp>
//#include <iostream>
//using namespace std;
//// Function to calculate Euclidean Distance
//double euclideanCompare(const vector<int>& a, const vector<int>& b) {
//	double sum = 0.0;
//	for (size_t i = 0; i < a.size(); i++) {
//		sum += pow(a[i] - b[i], 2);
//	}
//	return sqrt(sum);
//}
//int main() {
//	// White pixel counts for each radius for 4 images
//	vector<int> img1 = { 5, 49, 163, 465, 776, 1082, 1412, 1559, 1586 };
//	vector<int> img2 = { 2, 61, 214, 501, 842, 1191, 1509, 1613, 1621 };
//	vector<int> img3 = { 5, 62, 206, 436, 746, 1035, 1296, 1441, 1545 };
//	vector<int> img4 = { 0, 55, 213, 497, 832, 1178, 1489, 1592, 1602 };
//
//	// Calculate distances
//	double dist12 = euclideanCompare(img1, img2);
//	double dist13 = euclideanCompare(img1, img3);
//	double dist14 = euclideanCompare(img1, img4);
//	double dist23 = euclideanCompare(img2, img3);
//	double dist24 = euclideanCompare(img2, img4);
//	double dist34 = euclideanCompare(img3, img4);
//
//	// Output the results
//	cout << "Compare Image 1 and Image 2: " << dist12 << endl;
//	cout << "Compare Image 1 and Image 3: " << dist13 << endl;
//	cout << "Compare Image 1 and Image 4: " << dist14 << endl;
//	cout << "Compare Image 2 and Image 3: " << dist23 << endl;
//	cout << "Compare Image 2 and Image 4: " << dist24 << endl;
//	cout << "Compare Image 3 and Image 4: " << dist34 << endl;
//
//	return 0;
//}





















//----------------------Example code to use Basler camera-----------------------------
//int main(int argc, char* argv[]) try
//{
//    // Declare depth colorizer for pretty visualization of depth data
//    rs2::colorizer color_map;
//
//    // Declare RealSense pipeline, encapsulating the actual device and sensors
//    rs2::pipeline pipe;
//    // Start streaming with default recommended configuration
//    pipe.start();
//
//    using namespace cv;
//    const auto window_name = "Display Image";
//    namedWindow(window_name, WINDOW_AUTOSIZE);
//
//    while (waitKey(1) < 0 && getWindowProperty(window_name, WND_PROP_AUTOSIZE) >= 0)
//    {
//        rs2::frameset data = pipe.wait_for_frames(); // Wait for next set of frames from the camera
//        rs2::frame depth = data.get_depth_frame().apply_filter(color_map);
//
//        // Query frame size (width and height)
//        const int w = depth.as<rs2::video_frame>().get_width();
//        const int h = depth.as<rs2::video_frame>().get_height();
//
//        // Create OpenCV matrix of size (w,h) from the colorized depth data
//        Mat image(Size(w, h), CV_8UC3, (void*)depth.get_data(), Mat::AUTO_STEP);
//
//        // Update the window with new data
//        imshow(window_name, image);
//    }
//
//    return EXIT_SUCCESS;
//}
//catch (const rs2::error& e)
//{
//    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
//    return EXIT_FAILURE;
//}
//catch (const std::exception& e)
//{
//    std::cerr << e.what() << std::endl;
//    return EXIT_FAILURE;
//}


//-------------------(BTN1)_bai 4-------------------
//#include <opencv2/opencv.hpp>   // Include OpenCV API
//#include <iostream>
//using namespace std;
//using namespace cv;
//int main() {
//	Mat backgroundImage = imread("canhtay.png");
//	if (backgroundImage.empty()) {
//		std::cerr << "Error: Could not load the background image." << std::endl;
//		return -1;  // Exit the program gracefully.
//	}
//	Mat currentImage = imread("hinhnen.png");
//	if (currentImage.empty()) {
//		std::cerr << "Error: Could not load the current image." << std::endl;
//		return -1;  // Exit the program gracefully.
//	}
//	cv::Mat diffImage;
//	cv::absdiff(backgroundImage, currentImage, diffImage);
//	cv::Mat foregroundMask = cv::Mat::zeros(diffImage.rows, diffImage.cols, CV_8UC1);  
//	Mat final;
//	float threshold = 30.0f;
//	float dist;
//	for (int j = 0; j < diffImage.rows; ++j)  
//		for (int i = 0; i < diffImage.cols; ++i)
//	{
//		cv::Vec3b pix = diffImage.at<cv::Vec3b>(j, i);
//		dist = (pix[0] * pix[0] + pix[1] * pix[1] + pix[2] * pix[2]);  dist = sqrt(dist);
//		if (dist > threshold)
//		{
//			foregroundMask.at<unsigned char>(j, i) = 255;
//		}
//	}
//	cv::cvtColor(diffImage, final, cv::COLOR_RGB2GRAY);  
//	cv::threshold(final, final, 0, 255, 0);  
//	imshow("Background", backgroundImage);  imshow("Current", currentImage);
//	imshow("Diff", diffImage);  
//	imshow("final", final);  waitKey(0);
//	return 0;
//}


////-------------------(BTN1)_câu a_bài 4------------------------
//#include <opencv2/opencv.hpp>
//#include <iostream>
//#include <cmath>
//using namespace cv;
//using namespace std;
//
//// Function to calculate Euclidean distance between two RGB pixels
//double euclideanDistance(Vec3b pixel1, Vec3b pixel2) {
//    return sqrt(pow(pixel1[0] - pixel2[0], 2) + pow(pixel1[1] - pixel2[1], 2) + pow(pixel1[2] - pixel2[2], 2));
//}
//
//int main() {
//    // Load the two images
//    Mat img1 = imread("hinhnen.png");  // Image without hand
//    Mat img2 = imread("canhtay.png");  // Image with hand
//
//    // Check if images are loaded successfully
//    if (img1.empty() || img2.empty()) {
//        cout << "Error loading images!" << endl;
//        return -1;
//    }
//
//    // Resize both images to the same size
//    Size size(256, 256); // Resize to 256x256 pixels
//    resize(img1, img1, size);
//    resize(img2, img2, size);
//
//    // Initialize variables for distance calculation
//    double totalDistance = 0;
//    int pixelCount = img1.rows * img1.cols;
//
//    // Loop through each pixel and calculate the Euclidean distance
//    for (int i = 0; i < img1.rows; i++) {
//        for (int j = 0; j < img1.cols; j++) {
//            Vec3b pixel1 = img1.at<Vec3b>(i, j);
//            Vec3b pixel2 = img2.at<Vec3b>(i, j);
//            totalDistance += euclideanDistance(pixel1, pixel2);
//        }
//    }
//
//    // Calculate average Euclidean distance
//    double averageDistance = totalDistance / pixelCount;
//
//    // Output the result
//    cout << "Average Euclidean Distance: " << averageDistance << endl;
//}


//--------------------(BTN1)_Full bài 4---------------------
//#include <opencv2/opencv.hpp>
//#include <iostream>
//#include <cmath>
//using namespace cv;
//using namespace std;
//
//// Function to calculate Euclidean distance between two RGB pixels
//double euclideanDistance(Vec3b pixel1, Vec3b pixel2) {
//    return sqrt(pow(pixel1[0] - pixel2[0], 2) + pow(pixel1[1] - pixel2[1], 2) + pow(pixel1[2] - pixel2[2], 2));
//}
//
//int main() {
//    // Load the two images
//    Mat img1 = imread("hinhnen.png");  // Image without hand
//    Mat img2 = imread("canhtay.png");  // Image with hand
//
//    // Check if images are loaded successfully
//    if (img1.empty() || img2.empty()) {
//        cout << "Error loading images!" << endl;
//        return -1;
//    }
//
//    // Resize both images to the same size
//    Size size(256, 256); // Resize to 256x256 pixels
//    resize(img1, img1, size);
//    resize(img2, img2, size);
//
//    // Initialize variables for distance calculation
//    double totalDistance = 0;
//    int pixelCount = img1.rows * img1.cols;
//
//    // Loop through each pixel and calculate the Euclidean distance
//    for (int i = 0; i < img1.rows; i++) {
//        for (int j = 0; j < img1.cols; j++) {
//            Vec3b pixel1 = img1.at<Vec3b>(i, j);
//            Vec3b pixel2 = img2.at<Vec3b>(i, j);
//            totalDistance += euclideanDistance(pixel1, pixel2);
//        }
//    }
//
//    // Calculate average Euclidean distance
//    double averageDistance = totalDistance / pixelCount;
//
//    // Output the result
//    cout << "Average Euclidean Distance: " << averageDistance << endl;
//
//    //-------------Continue the code from part a---------------------
//    //Create a mask to highlight differences
//    Mat mask = Mat::zeros(img1.size(), CV_8UC1);
//
//    // Set threshold for Euclidean distance
//    double threshold = 30.0f;  // Adjust based on experiment
//
//    // Loop through each pixel and calculate the Euclidean distance
//    for (int i = 0; i < img1.rows; i++) {
//        for (int j = 0; j < img1.cols; j++) {
//            Vec3b pixel1 = img1.at<Vec3b>(i, j);
//            Vec3b pixel2 = img2.at<Vec3b>(i, j);
//
//            // Calculate Euclidean distance between corresponding pixels
//            double distance = euclideanDistance(pixel1, pixel2);
//
//            // If distance exceeds the threshold, mark it in the mask
//            if (distance > threshold) {
//                mask.at<uchar>(i, j) = 255;  // White to indicate the difference
//            }
//        }
//    }
//
//    // Show the mask highlighting detected changes (e.g., hand)
//    imshow("Detected Hand", mask);
//    waitKey(0);  // Wait for a key press
//
//    // Save the mask as an image
//    imwrite("detected_hand.png", mask);
//
//    return 0;
//}


//----------------------Example code to use Basler camera-----------------------------
//int main(int argc, char* argv[]) try
//{
//    // Declare depth colorizer for pretty visualization of depth data
//    rs2::colorizer color_map;
//
//    // Declare RealSense pipeline, encapsulating the actual device and sensors
//    rs2::pipeline pipe;
//    // Start streaming with default recommended configuration
//    pipe.start();
//
//    using namespace cv;
//    const auto window_name = "Display Image";
//    namedWindow(window_name, WINDOW_AUTOSIZE);
//
//    while (waitKey(1) < 0 && getWindowProperty(window_name, WND_PROP_AUTOSIZE) >= 0)
//    {
//        rs2::frameset data = pipe.wait_for_frames(); // Wait for next set of frames from the camera
//        rs2::frame depth = data.get_depth_frame().apply_filter(color_map);
//
//        // Query frame size (width and height)
//        const int w = depth.as<rs2::video_frame>().get_width();
//        const int h = depth.as<rs2::video_frame>().get_height();
//
//        // Create OpenCV matrix of size (w,h) from the colorized depth data
//        Mat image(Size(w, h), CV_8UC3, (void*)depth.get_data(), Mat::AUTO_STEP);
//
//        // Update the window with new data
//        imshow(window_name, image);
//    }
//
//    return EXIT_SUCCESS;
//}
//catch (const rs2::error& e)
//{
//    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
//    return EXIT_FAILURE;
//}
//catch (const std::exception& e)
//{
//    std::cerr << e.what() << std::endl;
//    return EXIT_FAILURE;
//}