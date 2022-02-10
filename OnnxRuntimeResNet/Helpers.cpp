#include "Helpers.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <array>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

static vector<float> loadImage(const string& filename, int sizeX = 224, int sizeY = 224)
{
    Mat image = imread(filename);
    if (image.empty()) {
        cout << "No image found.";
    }

    // convert from BGR to RGB
    cvtColor(image, image, COLOR_BGR2RGB);

    // resize
    resize(image, image, Size(sizeX, sizeY));

    // reshape to 1D
    image = image.reshape(1, 1);

    // uint_8, [0, 255] -> float, [0, 1]
    // Normailze number to between 0 and 1
    // Convert to vector<float> from cv::Mat.
    vector<float> vec;
    image.convertTo(vec, CV_32FC1, 1. / 255);

    // Transpose (Height, Width, Channel)(224,224,3) to (Chanel, Height, Width)(3,224,224)
    vector<float> output;
    for (size_t ch = 0; ch < 3; ++ch) {
        for (size_t i = ch; i < vec.size(); i += 3) {
            output.emplace_back(vec[i]);
        }
    }
    return output;
}

static vector<string> loadLabels(const string& filename)
{
    vector<string> output;

    ifstream file(filename);
    if (file) {
        string s;
        while (getline(file, s)) {
            output.emplace_back(s);
        }
        file.close();
    }

    return output;
}
