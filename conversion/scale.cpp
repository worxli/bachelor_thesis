#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <Eigen/Geometry>

using namespace std;
using namespace cv;


int main(int argc, char** argv)
{

    double sum1 = 0;
    double sum2 = 0;
    int line = 5;

    while(true)
        {

            std::stringstream depthName, mappedName;
            depthName << "/home/worxli/Datasets/data/dense/depth" << line << ".png";
            mappedName << "/home/worxli/Datasets/data/dense/mapped" << line << ".png";
            cv::Mat depth = cv::imread(depthName.str().c_str(), CV_LOAD_IMAGE_UNCHANGED);
            cv::Mat mapped = cv::imread(mappedName.str().c_str(), CV_LOAD_IMAGE_UNCHANGED);

            depth = depth * 5;

            // stop if empty matrices
            if (depth.rows == 0 || mapped.rows == 0)
                break;

            // loop over all pixels
            for (int i = 0; i < depth.cols; ++i)
            {
                for (int j = 0; j < depth.rows; ++j)
                {
                    sum1 =  sum1 + (double)depth.at<uint16_t>(j, i)/1000;
                    sum2 =  sum2 + (double)mapped.at<uint16_t>(j, i)/1000;
                }
            }

            cout << "sums" << endl;
            cout << sum1 << endl;
            cout << sum2 << endl;
            cout << sum2/sum1 << endl;
            
            line++;

        }

    cout << "end rsult" << endl;
    cout << sum2/sum1 << endl;
    cout << sum1/sum2 << endl;

    return 0;
}