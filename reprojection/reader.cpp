#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <cctype>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

using namespace cv;
using namespace std;

int main( int argc, char** argv ) {

  double scaling = 10;

  // camera calibration results
  Mat camera1, camera2, R, T;

  std::string file = "xmls/calib.xml";
  FileStorage cal(file, FileStorage::READ );
  cal["camera1"] >> camera1;
  cal["camera2"] >> camera2;
  cal["R"] >> R;
  cal["T"] >> T;

  Mat rgb, depthRaw, depth, mapped, depthScale, mappedScale;
  
  float *ptrOrig;
  uint16_t *ptrDepth, *ptrMapped, *ptrDepthScale, *ptrMappedScale;

  Size imgSize;

  vector<int> compression_params;
  compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
  compression_params.push_back(1);


  for (int k = 0;;++k)
  {

    std::string file = "xmls/sequence" + std::to_string (k) + ".xml";
    std::string rgbName = "xmls/sequence" + std::to_string (k) + ".png";

    FileStorage fs(file, FileStorage::READ );
    // fs["image"] >> rgb;
    fs["depth"] >> depthRaw;
    rgb = imread(rgbName, CV_LOAD_IMAGE_UNCHANGED);

    if(rgb.cols == 0) {
      cout << k << " pairs saved" << endl;
      break;
    }

    cout << "reading image: " << k << endl;

    // cvtColor(rgb, rgb, CV_BGR2RGB);

    imgSize = rgb.size();
    cout << imgSize << endl;
    // cout << depthRaw.size() << endl;
    //resize(depthRaw, depthRaw, imgSize);

    // new empty mats
    depth = Mat::ones(imgSize, CV_16UC1);
    mapped = Mat::zeros(imgSize, CV_16UC1);
    depthScale = Mat::ones(imgSize, CV_16UC1);
    mappedScale = Mat::zeros(imgSize, CV_16UC1);

    // orig pmat pointer
    ptrOrig = (float*)(depthRaw.data);

    // pointer to new elems
    ptrDepth = (uint16_t*)(depth.data);
    ptrMapped = (uint16_t*)(mapped.data);
    ptrDepthScale = (uint16_t*)(depthScale.data);
    ptrMappedScale = (uint16_t*)(mappedScale.data);

    for (int row = 0; row < depthRaw.rows; ++row)
    {
      // for (int col = 0; col < depthRaw.cols/3; ++col)
      for (int col = 0; col < depthRaw.cols; ++col)
      {

        // not mapped depth image
        ptrDepth[depthRaw.cols * row + col ] = (uint16_t)ptrOrig[depthRaw.cols * row + col];
        // ptrDepthScale[depthRaw.cols * row + col ] = (uint16_t)ptrOrig[depthRaw.cols * row + col] * scaling;       
            
        // get depth for remapping
        double depthVal = (double)ptrOrig[depthRaw.cols*row + col];
        // if (depthVal != depthVal)
        // {
        //   // depthVal = 10;
        // }
            
        // Map depthcam depth to 3D point
        Mat_<double> P3D = Mat_<double>(3,1);

        P3D(0) = (col - camera1.at<double>(0,2)) * depthVal / camera1.at<double>(0,0);
        P3D(1) = (row - camera1.at<double>(1,2)) * depthVal / camera1.at<double>(1,1);
        P3D(2) = depthVal;
            
        // Rotate and translate 3D point
        Mat_<double> P3Dp;
        P3Dp = (R*P3D) + T;
            
        // Project 3D point to rgbcam
        double xrgb = (P3Dp(0) * camera2.at<double>(0,0) / P3Dp(2)) + camera2.at<double>(0,2);
        double yrgb = (P3Dp(1) * camera2.at<double>(1,1) / P3Dp(2)) + camera2.at<double>(1,2);
        double nDepth = P3Dp(2);  

        // "Interpolate" pixel coordinates (Nearest Neighbors)
        int px_rgbcam = cvRound(xrgb);
        int py_rgbcam = cvRound(yrgb);

        // cout << "image " << k << " y: " << row << " x: " << col << endl;
        // cout << "is mapped to " <<  "y: " << yrgb << " x: " << xrgb << endl;
        // cout << "old depth " << depthVal << " new depth " << nDepth << endl;
        // cout << endl;

        // Handle 3D occlusions
        uint16_t &depth_rgbcam = ptrMapped[depthRaw.cols * py_rgbcam + px_rgbcam];
        uint16_t &depth_rgbcamScale = ptrMappedScale[depthRaw.cols * py_rgbcam + px_rgbcam];

        if(px_rgbcam - depthRaw.cols < 0 && py_rgbcam - depthRaw.rows < 0)
        {

          if(depth_rgbcam == 0 || (uint16_t)nDepth < depth_rgbcam) 
          {
            depth_rgbcam = 5*(uint16_t)nDepth;
          }   
        } 
                

        // if(depth_rgbcamScale == 0 || (uint16_t)nDepth < depth_rgbcamScale) {
        //   // depth_rgbcamScale = (uint16_t)(nDepth * scaling);
        // } 
      }
    }

    // imshow("depth" + std::to_string (k) + "",depth);
    // std::cin.ignore();
    // imshow("depthMapped" + std::to_string (k) + "",mapped);
    // std::cin.ignore();

    try {
      imwrite("data/img/color" + std::to_string (k) + ".png", rgb, compression_params);
    } catch (runtime_error& ex) {
      fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
      return 1;
    }

        try {
      imwrite("data/img/color" + std::to_string (k) + ".jpg", rgb);
    } catch (runtime_error& ex) {
      fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
      return 1;
    }

    // try {
    //   imwrite("data/depth/depth" + std::to_string (k) + ".png", depth, compression_params);
    // } catch (runtime_error& ex) {
    //   fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
    //   return 1;
    // }

    try {
      imwrite("data/depth/mapped" + std::to_string (k) + ".png", mapped, compression_params);
    } catch (runtime_error& ex) {
      fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
      return 1;
    }

    // try {
    //   imwrite("data/depth/depthScale" + std::to_string (k) + ".png", depthScale, compression_params);
    // } catch (runtime_error& ex) {
    //   fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
    //   return 1;
    // }

    // try {
    //   imwrite("data/depth/mappedScale" + std::to_string (k) + ".png", mappedScale, compression_params);
    // } catch (runtime_error& ex) {
    //   fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
    //   return 1;
    // }

  }

  return 0;

}