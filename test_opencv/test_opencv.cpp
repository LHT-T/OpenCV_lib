// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

//#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
//#include <opencv2/opencv.hpp>   // Include OpenCV API
//#include <iostream>
//using namespace std;
//using namespace cv;

////------------------------(BTN1)_Ex2_part b--------------------------
//#include <opencv2/opencv.hpp>   // Include OpenCV API
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
//------------------------------------------------------------------

//-------------------(BTN1)_Ex2_part c---------------------------------
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
//-----------------------------------------------------------------------

//-----------------(BTN1)_Ex3_Full------------------------------
//#include <opencv2/opencv.hpp>
//#include <iostream>
//#include <cmath>
//
//using namespace std;
//using namespace cv;
//
//int main() {
//    Mat backgroundImage = imread("canhtay.png"); //Image with hand
//    Mat currentImage = imread("hinhnen.png"); //Image without hand
//    
//    // Check if images are loaded successfully
//    if (backgroundImage.empty() || currentImage.empty()) {
//        cout << "Error loading images!" << endl;
//        return -1;
//    }
//
//    // Resize both image to the same size
//    if (backgroundImage.size() != currentImage.size()) {
//        resize(currentImage, currentImage, backgroundImage.size());
//    }
//
//    //Reducing blur
//    GaussianBlur(backgroundImage, backgroundImage, cv::Size(5, 5), 0);
//    GaussianBlur(currentImage, currentImage, cv::Size(5, 5), 0);
//
//    Mat diffImage;
//    absdiff(backgroundImage, currentImage, diffImage);
//
//    // Create a foreground mask
//    Mat foregroundMask = cv::Mat::zeros(diffImage.rows, diffImage.cols, CV_8UC1);
//    float threshold = 30.0f;
//    float dist;
//
//    for (int j = 0; j < diffImage.rows; ++j) {
//        for (int i = 0; i < diffImage.cols; ++i) {
//            Vec3b pix = diffImage.at<Vec3b>(j, i);
//            dist = sqrt(pix[0] * pix[0] + pix[1] * pix[1] + pix[2] * pix[2]);
//
//            if (dist > threshold) {
//                foregroundMask.at<unsigned char>(j, i) = 255;
//            }
//        }
//    }
//
//    cout << "Calculated Euclidean distance between two (R,G,B) of pictures= " << dist;
//    imshow("Detected hand", foregroundMask);
//    waitKey(0);
//}
//---------------------------------------------------------------------------------

//----------------(BTN1)_Ex4_part a---------------------------------
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
//----------------------------------------------------------------------------



//------------------(BTN1)_Ex4_part b-----------------------------------------
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
//---------------------------------------------------------------------------------------

//----------------(BTN1)_Ex5_Full------------------------------------------------
//#include <opencv2/opencv.hpp>
//#include <iostream>
//#include <iomanip>
//
//using namespace cv;
//
//int main() {
//    Mat img1 = imread("Picture1.png", IMREAD_GRAYSCALE);
//    Mat img2 = imread("Picture2.png", IMREAD_GRAYSCALE);
//    Mat img3 = imread("Picture3.png", IMREAD_GRAYSCALE);
//    Mat img4 = imread("Picture4.png", IMREAD_GRAYSCALE);
//
//    threshold(img1, img1, 128, 255, THRESH_BINARY);
//    threshold(img2, img2, 128, 255, THRESH_BINARY);
//    threshold(img3, img3, 128, 255, THRESH_BINARY);
//    threshold(img4, img4, 128, 255, THRESH_BINARY);
//
//    int r[9] = { 1, 5, 10, 15, 20, 25, 30, 35, 40 };
//    int white_count[4][9] = { 0 };
//    Mat display_images[] = { img1.clone(), img2.clone(), img3.clone(), img4.clone() };
//
//    Mat images[] = { img1, img2, img3, img4 };
//    for (int imgIndex = 0; imgIndex < 4; imgIndex++) {
//        for (int i = 0; i < 9; i++) {
//            int count_white = 0;
//
//            for (float phi = 0; phi < (2 * CV_PI); phi += 0.01) {
//                int x = 100 - static_cast<int>(r[i] * cos(phi));
//                int y = 80 - static_cast<int>(r[i] * sin(phi));
//
//                if (x >= 0 && x < images[imgIndex].cols && y >= 0 && y < images[imgIndex].rows) {
//                    if (images[imgIndex].at<uchar>(y, x) == 255)
//                        count_white++;
//                }
//                // Draw circle on display image                    
//                circle(display_images[imgIndex], Point(x, y), 1, Scalar(255, 0, 0), -1); // Draw a small circle
//            }
//            white_count[imgIndex][i] = count_white;
//            std::cout << "Image " << (imgIndex + 1) << " - Radius r = " << r[i] << " - White Pixels: " << count_white << std::endl;
//        }
//    }
//
//    for (int n = 0; n < 4; n++) {
//        for (int m = n + 1; m < 4; m++) {
//            float error = 0.0;
//            int validComparisons = 0;
//
//            for (int i = 0; i < 9; i++) {
//                if (white_count[m][i] > 0) {
//                    float diff = (float)white_count[n][i] / (float)white_count[m][i];
//                    error += (1.0 - diff) * (1.0 - diff);
//                    validComparisons++;
//                }
//            }
//            if (validComparisons > 0) {
//                error /= validComparisons;
//            }
//            std::cout << "Error between image " << (n + 1) << " and image " << (m + 1) << ": "
//                << std::fixed << std::setprecision(2) << (error * 100) << "%" << std::endl;
//        }
//    }
//    //Display the images with circles drawn
//    for (int i = 0; i < 4; i++) {
//        // Convert to BGR for display
//        Mat color_image;
//        cvtColor(display_images[i], color_image, COLOR_GRAY2BGR);
//
//        // Show the image in a window
//        std::string window_name = "Image " + std::to_string(i + 1);
//        imshow(window_name, color_image);
//
//        // Save the images with circles
//        imwrite("ImageWithCircles_" + std::to_string(i + 1) + ".png", color_image);
//    }
//
//    waitKey(0); // Wait for a key press to close the windows
//    return 0;
//}
//----------------------------------------------------------------------------
// 
// 
// 
// 
// 
//-------------(BTN1)_Ex5_concentric circles but wrong answer------------------
//#include <opencv2/opencv.hpp>
//#include <iostream>
//#include <iomanip> // For std::setprecision
//
//using namespace cv;
//
//int main() {
//    // Load images in grayscale
//    Mat img1 = imread("Picture1.png", IMREAD_GRAYSCALE);
//    Mat img2 = imread("Picture2.png", IMREAD_GRAYSCALE);
//    Mat img3 = imread("Picture3.png", IMREAD_GRAYSCALE);
//    Mat img4 = imread("Picture4.png", IMREAD_GRAYSCALE);
//
//    // Threshold each image
//    threshold(img1, img1, 128, 255, THRESH_BINARY);
//    threshold(img2, img2, 128, 255, THRESH_BINARY);
//    threshold(img3, img3, 128, 255, THRESH_BINARY);
//    threshold(img4, img4, 128, 255, THRESH_BINARY);
//
//    // Radii for circles
//    int r[9] = { 1, 5, 10, 15, 20, 25, 30, 35, 40 };
//    int white_count[4][9] = { 0 }; // Store counts for 4 images
//
//    // Create a copy of original images for drawing circles
//    Mat display_images[] = { img1.clone(), img2.clone(), img3.clone(), img4.clone() };
//
//    // Process each image
//    Mat images[] = { img1, img2, img3, img4 };
//    for (int imgIndex = 0; imgIndex < 4; imgIndex++) {
//        for (int i = 0; i < 9; i++) {
//            int count_white = 0;
//
//            // Loop through angles for circle points
//            for (float phi = 0; phi < (2 * CV_PI); phi += 0.025) {
//                int x = 100 - static_cast<int>(r[i] * cos(phi));
//                int y = 100 - static_cast<int>(r[i] * sin(phi));
//
//                // Ensure the point is within the image bounds
//                if (x >= 0 && x < images[imgIndex].cols && y >= 0 && y < images[imgIndex].rows) {
//                    // Check if the pixel is white
//                    if (images[imgIndex].at<uchar>(y, x) == 255) {
//                        count_white++;
//                    }
//                    // Draw circle on display image
//                    circle(display_images[imgIndex], Point(x, y), 1, Scalar(255, 0, 0), -1); // Draw a small circle
//                }
//            }
//            white_count[imgIndex][i] = count_white; // Store count for the current image
//            std::cout << "Image " << (imgIndex + 1) << " - Radius r = " << r[i] << " - White Pixels: " << count_white << std::endl;
//        }
//    }
//
//    // Compare images in pairs and calculate errors
//    for (int n = 0; n < 4; n++) {
//        for (int m = n + 1; m < 4; m++) {
//            float error = 0.0;
//            int validComparisons = 0; // Counter for valid comparisons
//
//            // Calculate error for the pixel counts between image n and m
//            for (int i = 0; i < 9; i++) {
//                if (white_count[m][i] > 0) { // Avoid division by zero
//                    float diff = (float)white_count[n][i] / (float)white_count[m][i];
//                    error += (1.0 - diff) * (1.0 - diff);
//                    validComparisons++; // Increment valid comparison count
//                }
//            }
//            if (validComparisons > 0) { // Ensure we divide only if we have valid comparisons
//                error /= validComparisons; // Average over valid comparisons
//            }
//
//            // Output the error
//            std::cout << "Error between image " << (n + 1) << " and image " << (m + 1) << ": "
//                << std::fixed << std::setprecision(2) << (error * 100) << "%" << std::endl;
//        }
//    }
//
//    // Display the images with circles drawn
//    for (int i = 0; i < 4; i++) {
//        // Convert to BGR for display
//        Mat color_image;
//        cvtColor(display_images[i], color_image, COLOR_GRAY2BGR);
//
//        // Show the image in a window
//        std::string window_name = "Image " + std::to_string(i + 1);
//        imshow(window_name, color_image);
//
//        // Save the images with circles
//        imwrite("ImageWithCircles_" + std::to_string(i + 1) + ".png", color_image);
//    }
//
//    waitKey(0); // Wait for a key press to close the windows
//    return 0;
//}
//--------------------------------------------------------------------------






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



//-----------------(BTN1)_Ex3_Full------------------------------
//#include <opencv2/opencv.hpp>
//#include <iostream>
//#include <cmath>
//
//using namespace std;
//using namespace cv;
//
//int main() {
//    Mat backgroundImage = imread("canhtay.png"); //Image with hand
//    Mat currentImage = imread("hinhnen.png"); //Image without hand
//    
//    // Check if images are loaded successfully
//    if (backgroundImage.empty() || currentImage.empty()) {
//        cout << "Error loading images!" << endl;
//        return -1;
//    }
//
//    // Resize both image to the same size
//    if (backgroundImage.size() != currentImage.size()) {
//        resize(currentImage, currentImage, backgroundImage.size());
//    }
//
//    //Reducing blur
//    GaussianBlur(backgroundImage, backgroundImage, cv::Size(5, 5), 0);
//    GaussianBlur(currentImage, currentImage, cv::Size(5, 5), 0);
//
//    Mat diffImage;
//    absdiff(backgroundImage, currentImage, diffImage);
//
//    // Create a foreground mask
//    Mat foregroundMask = cv::Mat::zeros(diffImage.rows, diffImage.cols, CV_8UC1);
//    float threshold = 30.0f;
//    float dist;
//
//    for (int j = 0; j < diffImage.rows; ++j) {
//        for (int i = 0; i < diffImage.cols; ++i) {
//            Vec3b pix = diffImage.at<Vec3b>(j, i);
//            dist = sqrt(pix[0] * pix[0] + pix[1] * pix[1] + pix[2] * pix[2]);
//
//            if (dist > threshold) {
//                foregroundMask.at<unsigned char>(j, i) = 255;
//            }
//        }
//    }
//
//    cout << "Calculated Euclidean distance between two (R,G,B) of pictures= " << dist;
//    imshow("Detected hand", foregroundMask);
//    waitKey(0);
//}
//---------------------------------------------------------------------------------


//----------------(BTN1)_Ex5_Full------------------------------------------------
//#include <opencv2/opencv.hpp>
//#include <iostream>
//#include <iomanip>
//
//using namespace cv;
//
//int main() {
//    Mat img1 = imread("Picture1.png", IMREAD_GRAYSCALE);
//    Mat img2 = imread("Picture2.png", IMREAD_GRAYSCALE);
//    Mat img3 = imread("Picture3.png", IMREAD_GRAYSCALE);
//    Mat img4 = imread("Picture4.png", IMREAD_GRAYSCALE);
//
//    threshold(img1, img1, 128, 255, THRESH_BINARY);
//    threshold(img2, img2, 128, 255, THRESH_BINARY);
//    threshold(img3, img3, 128, 255, THRESH_BINARY);
//    threshold(img4, img4, 128, 255, THRESH_BINARY);
//
//    int r[9] = { 1, 5, 10, 15, 20, 25, 30, 35, 40 };
//    int white_count[4][9] = { 0 };
//    Mat display_images[] = { img1.clone(), img2.clone(), img3.clone(), img4.clone() };
//
//    Mat images[] = { img1, img2, img3, img4 };
//    for (int imgIndex = 0; imgIndex < 4; imgIndex++) {
//        for (int i = 0; i < 9; i++) {
//            int count_white = 0;
//
//            for (float phi = 0; phi < (2 * CV_PI); phi += 0.01) {
//                int x = 100 - static_cast<int>(r[i] * cos(phi));
//                int y = 80 - static_cast<int>(r[i] * sin(phi));
//
//                if (x >= 0 && x < images[imgIndex].cols && y >= 0 && y < images[imgIndex].rows) {
//                    if (images[imgIndex].at<uchar>(y, x) == 255)
//                        count_white++;
//                }
//                // Draw circle on display image                    
//                circle(display_images[imgIndex], Point(x, y), 1, Scalar(255, 0, 0), -1); // Draw a small circle
//            }
//            white_count[imgIndex][i] = count_white;
//            std::cout << "Image " << (imgIndex + 1) << " - Radius r = " << r[i] << " - White Pixels: " << count_white << std::endl;
//        }
//    }
//
//    for (int n = 0; n < 4; n++) {
//        for (int m = n + 1; m < 4; m++) {
//            float error = 0.0;
//            int validComparisons = 0;
//
//            for (int i = 0; i < 9; i++) {
//                if (white_count[m][i] > 0) {
//                    float diff = (float)white_count[n][i] / (float)white_count[m][i];
//                    error += (1.0 - diff) * (1.0 - diff);
//                    validComparisons++;
//                }
//            }
//            if (validComparisons > 0) {
//                error /= validComparisons;
//            }
//            std::cout << "Error between image " << (n + 1) << " and image " << (m + 1) << ": "
//                << std::fixed << std::setprecision(2) << (error * 100) << "%" << std::endl;
//        }
//    }
//    //Display the images with circles drawn
//    for (int i = 0; i < 4; i++) {
//        // Convert to BGR for display
//        Mat color_image;
//        cvtColor(display_images[i], color_image, COLOR_GRAY2BGR);
//
//        // Show the image in a window
//        std::string window_name = "Image " + std::to_string(i + 1);
//        imshow(window_name, color_image);
//
//        // Save the images with circles
//        imwrite("ImageWithCircles_" + std::to_string(i + 1) + ".png", color_image);
//    }
//
//    waitKey(0); // Wait for a key press to close the windows
//    return 0;
//}
//----------------------------------------------------------------------------
// 
// 
// 
// 
// 
//-------------(BTN1)_Ex5_concentric circles but wrong answer------------------
//#include <opencv2/opencv.hpp>
//#include <iostream>
//#include <iomanip> // For std::setprecision
//
//using namespace cv;
//
//int main() {
//    // Load images in grayscale
//    Mat img1 = imread("Picture1.png", IMREAD_GRAYSCALE);
//    Mat img2 = imread("Picture2.png", IMREAD_GRAYSCALE);
//    Mat img3 = imread("Picture3.png", IMREAD_GRAYSCALE);
//    Mat img4 = imread("Picture4.png", IMREAD_GRAYSCALE);
//
//    // Threshold each image
//    threshold(img1, img1, 128, 255, THRESH_BINARY);
//    threshold(img2, img2, 128, 255, THRESH_BINARY);
//    threshold(img3, img3, 128, 255, THRESH_BINARY);
//    threshold(img4, img4, 128, 255, THRESH_BINARY);
//
//    // Radii for circles
//    int r[9] = { 1, 5, 10, 15, 20, 25, 30, 35, 40 };
//    int white_count[4][9] = { 0 }; // Store counts for 4 images
//
//    // Create a copy of original images for drawing circles
//    Mat display_images[] = { img1.clone(), img2.clone(), img3.clone(), img4.clone() };
//
//    // Process each image
//    Mat images[] = { img1, img2, img3, img4 };
//    for (int imgIndex = 0; imgIndex < 4; imgIndex++) {
//        for (int i = 0; i < 9; i++) {
//            int count_white = 0;
//
//            // Loop through angles for circle points
//            for (float phi = 0; phi < (2 * CV_PI); phi += 0.025) {
//                int x = 100 - static_cast<int>(r[i] * cos(phi));
//                int y = 100 - static_cast<int>(r[i] * sin(phi));
//
//                // Ensure the point is within the image bounds
//                if (x >= 0 && x < images[imgIndex].cols && y >= 0 && y < images[imgIndex].rows) {
//                    // Check if the pixel is white
//                    if (images[imgIndex].at<uchar>(y, x) == 255) {
//                        count_white++;
//                    }
//                    // Draw circle on display image
//                    circle(display_images[imgIndex], Point(x, y), 1, Scalar(255, 0, 0), -1); // Draw a small circle
//                }
//            }
//            white_count[imgIndex][i] = count_white; // Store count for the current image
//            std::cout << "Image " << (imgIndex + 1) << " - Radius r = " << r[i] << " - White Pixels: " << count_white << std::endl;
//        }
//    }
//
//    // Compare images in pairs and calculate errors
//    for (int n = 0; n < 4; n++) {
//        for (int m = n + 1; m < 4; m++) {
//            float error = 0.0;
//            int validComparisons = 0; // Counter for valid comparisons
//
//            // Calculate error for the pixel counts between image n and m
//            for (int i = 0; i < 9; i++) {
//                if (white_count[m][i] > 0) { // Avoid division by zero
//                    float diff = (float)white_count[n][i] / (float)white_count[m][i];
//                    error += (1.0 - diff) * (1.0 - diff);
//                    validComparisons++; // Increment valid comparison count
//                }
//            }
//            if (validComparisons > 0) { // Ensure we divide only if we have valid comparisons
//                error /= validComparisons; // Average over valid comparisons
//            }
//
//            // Output the error
//            std::cout << "Error between image " << (n + 1) << " and image " << (m + 1) << ": "
//                << std::fixed << std::setprecision(2) << (error * 100) << "%" << std::endl;
//        }
//    }
//
//    // Display the images with circles drawn
//    for (int i = 0; i < 4; i++) {
//        // Convert to BGR for display
//        Mat color_image;
//        cvtColor(display_images[i], color_image, COLOR_GRAY2BGR);
//
//        // Show the image in a window
//        std::string window_name = "Image " + std::to_string(i + 1);
//        imshow(window_name, color_image);
//
//        // Save the images with circles
//        imwrite("ImageWithCircles_" + std::to_string(i + 1) + ".png", color_image);
//    }
//
//    waitKey(0); // Wait for a key press to close the windows
//    return 0;
//}
//--------------------------------------------------------------------------






