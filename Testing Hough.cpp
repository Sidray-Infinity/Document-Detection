#include <opencv2/opencv.hpp>
#include <opencv2/core/types_c.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/mat.hpp>
#include <iostream>
#include <typeinfo>
#include <vector>
#include <algorithm>
#include <iterator>
#include <string>

using namespace std;
using namespace cv;

float RAD_LOW = 0.0872665; // 5 deg
float RAD_LOW_ACUTE = 1.48353;  // 85 deg
float RAD_HIGH_ACUTE = 1.65806; // 95 deg
float RAD_LOW_OBTUSE = 3.05433;  // 175 deg
float RAD_ISO_LOW = 1.39626;  // 80 deg
float RAD_ISO_HIGH = 1.74533; // 100 deg
double RATIO_LOW = 1.18;
double RATIO_HIGH = 1.45;

Mat print_lines(Mat img, vector<Vec2f> lines) {
	Mat frame = img.clone();

	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(frame, pt1, pt2, Scalar(0,0,255), 2, LINE_AA);
	}
	return frame;
}

double det_angle(Vec<float, 2> a, Vec<float, 2> b) {
	return abs(a[1] - b[1]);
}

int searchLine(vector<Vec2f> set, Vec<float, 2> line) {
	double rho = line[0];
	double theta = line[1];

	try {
		for (int i = 0; i < set.size(); i++) {
			if ((set[i][0] == rho) && (set[i][1] == theta))
				return 0;
		}
		return 1;
	}
	catch (...) {
		return 1;
	}
}

vector<Vec2f> isolate_lines(vector<Vec2f> lines) {
	vector<Vec2f> new_lines{};
	try {
		for (int i = 0; i < lines.size(); i++) {
			for (int j = 0; j < lines.size(); j++) {
				double angle = det_angle(lines[i], lines[j]);
				if ((angle > RAD_ISO_LOW) && (angle < RAD_ISO_HIGH)) {
					if (searchLine(new_lines, lines[i]))
						new_lines.push_back(lines[i]);
					if (searchLine(new_lines, lines[j]))
						new_lines.push_back(lines[j]);
				}
			}
		}
		return new_lines;
	}
	catch (...) {
		return new_lines;
	}
}

void seg_hor_ver(vector<Vec2f> lines, vector<Vec2f> &hor, vector<Vec2f> &ver) {
	try {
		for (int i = 0; i < lines.size(); i++) {
			double theta = lines[i][1];
			if ((theta < RAD_LOW) || (theta > RAD_LOW_OBTUSE))
				ver.push_back(lines[i]);
			if ((theta > RAD_LOW_ACUTE) && (theta < RAD_HIGH_ACUTE))
				hor.push_back(lines[i]);
		}
	}
	catch (...) {
		return;
	}
}

bool compareContourAreas(vector<Point> contour1, vector<Point> contour2) {
	double i = fabs(contourArea(cv::Mat(contour1)));
	double j = fabs(contourArea(cv::Mat(contour2)));
	return (i > j);
}

bool compareRho(Vec<float, 2> line1, Vec<float, 2> line2) {
	double i = line1[0];
	double j = line2[0];
	return i < j;
}

bool cal_aspect_ratio(vector<Point> points) {
	float w1 = abs(points[2].x - points[1].x);
	float w2 = abs(points[3].x - points[0].x);
	float w = min(w1, w2);

	float h1 = abs(points[0].y - points[1].y);
	float h2 = abs(points[2].y - points[3].y);
	float h = min(h1, h2);

	float ratio = (float)w / h;


	if (ratio > RATIO_LOW && ratio < RATIO_HIGH) {
		cout << "RATIO: " << ratio << '\n';
		return true;
	}
	return false;
}


Mat extract_doc(Mat image, vector<Point> contour) {
	Mat doc;
	Rect rect = boundingRect(contour);
	doc = image(rect);
	return doc;
}

Point intersectionPoint(Vec<float, 2> line1, Vec<float, 2> line2) {
	Point pt;
	double rho1 = line1[0];
	double rho2 = line2[0];
	double theta1 = line1[1];
	double theta2 = line2[1];
	double min_thresh_dem = 0.01;

	double dem_y = (double)sin(theta2) * cos(theta1) - (double)sin(theta1) * cos(theta2);
	double num_y = (double)rho2 * cos(theta1) - (double)rho1 * cos(theta2);
	double dem_x = cos(theta1);

	if (dem_y == 0)
		dem_y = min_thresh_dem;
	if (dem_x == 0)
		dem_x = min_thresh_dem;

	//cout << "DEM_Y: " << dem_y << '\n';
	//cout << "X DEM: " << dem_x << '\n';

	pt.y = (double)num_y / dem_y;
	pt.x = (double)(rho1 - pt.y * sin(theta1)) / dem_x;

	return pt;
}
 
double polygonArea(vector<Point> points) {
	double area = 0.0;
	int j = points.size()-1;
	for (int i = 0; i < points.size(); i++) {
		area += ((double)points[i].x + points[j].x) *((double)points[i].y - points[j].y);
		j=i;
	}
	return abs(area / 2);
}


int main() {

	VideoCapture cap(0);
	if (!cap.isOpened())  // check if we succeeded
		return -1;

	Mat edges;
	int count = 0;

	while(true) {

		Mat frame;
		cap >> frame; // get a new frame from camera

		int x = 150;
		int y = 60;
		int h = 181;
		int w = 427;

		int frame_area = w * h;
		frame = frame(Rect(x, y, w, h));
		imshow("Orig", frame);

		cvtColor(frame, edges, COLOR_BGR2GRAY);
		GaussianBlur(edges, edges, Size(5, 5), 0);
		Canny(edges, edges, 20, 30);

		imshow("canny", edges);

		//imshow("Canny", edges);

		vector<Vec2f> lines, new_lines, hor_lines, ver_lines;

		HoughLines(edges, lines, 1, CV_PI / 180, 90);

		try {
			new_lines = isolate_lines(lines);
			seg_hor_ver(new_lines, hor_lines, ver_lines);

			if (hor_lines.size() != 0 && ver_lines.size() != 0) {

				sort(hor_lines.begin(), hor_lines.end(), compareRho);
				sort(ver_lines.begin(), ver_lines.end(), compareRho);

				vector<Vec2f> final_lines;
				final_lines.push_back(hor_lines[0]);
				final_lines.push_back(ver_lines[ver_lines.size() - 1]);
				final_lines.push_back(hor_lines[hor_lines.size() - 1]);
				final_lines.push_back(ver_lines[0]);

				Mat fin_lines_frame = print_lines(frame, final_lines);
				imshow("Final Lines", fin_lines_frame);

				if (final_lines.size() == 4) {
					vector<Point> points;
					//vector<Vec<double, 2>> points;
					for (int i = 0; i < final_lines.size() - 1; i++) {
						int j = i + 1;
						Point x = intersectionPoint(final_lines[i], final_lines[j]);
						points.push_back(x);
					}
					Point x = intersectionPoint(final_lines[3], final_lines[0]);
					points.push_back(x);

					double poly_area = polygonArea(points);
					
					int flag = 1;
					for (int i = 0; i < points.size(); i++)
						if (points[i].x == 0 || points[i].y == 0)
							flag = 0;

					if (poly_area != 0.0 && flag != 0) {

						cout << "POINTS: \n";
						for (int i = 0; i < points.size(); i++)
							cout << points[i].x << ", " << points[i].y << '\n';
						cout << "------------------------\n";

						if (cal_aspect_ratio(points)) {
							if ((double)poly_area / frame_area >= 0.50) {
								int x = points[3].x;
								int y = points[3].y;
								int w1 = abs(points[0].x - points[3].x);
								int w2 = abs(points[1].x - points[2].x);
								int w = min(w1, w2);
								int h1 = abs(points[0].y - points[1].y);
								int h2 = abs(points[3].y - points[2].y);
								int h = min(h1, h2);

								cout << "x: " << x << '\n';
								cout << "y: " << y << '\n';
								cout << "w: " << w << '\n';
								cout << "h: " << h << '\n';
								cout << "------------------------\n";

								Rect R = Rect(x, y, w, h);

								Mat doc = frame(R);
								imshow("DOC", doc);

								polylines(frame, points, true, Scalar(0, 255, 0), 2);
							}
							else {
								polylines(frame, points, true, Scalar(0, 0, 255), 2);
								cout << "Bring it closer.\n";
							}
						}
					}
				}
			}
			imshow("Final", frame);
		}
		catch (...) {
			cout << "IN CATCH\n";
			continue;
		}

		if (waitKey(30) >= 0) break;
	}
	cap.release();
	destroyAllWindows();
	return 0;
}
