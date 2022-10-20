

#include <filesystem>
#include <fstream>
#include <iostream>
#include <array>

#include "Helpers.h"

#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>


static std::vector<float> loadImage(const std::string& filename, int sizeX = 224, int sizeY = 224)
{
    cv::Mat image = cv::imread(filename);
    if (image.empty()) {
        std::cout << "No image found.";
    }

    // convert from BGR to RGB
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    // resize
    cv::resize(image, image, cv::Size(sizeX, sizeY));

    // reshape to 1D
    image = image.reshape(1, 1);

    // uint_8, [0, 255] -> float, [0, 1]
    // Normailze number to between 0 and 1
    // Convert to vector<float> from cv::Mat.
    std::vector<float> vec;
    image.convertTo(vec, CV_32FC1, 1. / 255);

    // Transpose (Height, Width, Channel)(224,224,3) to (Chanel, Height, Width)(3,224,224)
    std::vector<float> output;
    for (size_t ch = 0; ch < 3; ++ch) {
        for (size_t i = ch; i < vec.size(); i += 3) {
            output.emplace_back(vec[i]);
        }
    }
    return output;
}

static std::vector<std::string> loadLabels(const std::string& filename)
{
    std::vector<std::string> output;

    std::ifstream file(filename);
    if (file) {
        std::string s;
        while (getline(file, s)) {
            output.emplace_back(s);
        }
        file.close();
    }

    return output;
}
