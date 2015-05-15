#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <Eigen/Geometry>

using namespace std;
using namespace cv;

struct Pose {

  Eigen::Matrix3d r;
  Eigen::Vector3d t;
  Eigen::Vector3d nt;
  Eigen::Matrix3d k;
  Eigen::Quaterniond q;
  Eigen::Quaterniond qinv;

  float fx, fy, cx, cy;

  void LoadStatic(float _fx, float _fy, float _cx, float _cy ) {
    fx = _fx;
    fy = _fy;
    cx = _cx;
    cy = _cy;
  }

  void Load(std::ifstream* stream) {

    float r11, r12, r13, r21, r22, r23, r31, r32, r33;
    float t1, t2, t3;
    float chunk;

    std::string line;
    getline(*stream,line);
    std::stringstream rotation(line);

    rotation >> r11;
    rotation >> r12;
    rotation >> r13;
    rotation >> r21;
    rotation >> r22;
    rotation >> r23;
    rotation >> r31;
    rotation >> r32;
    rotation >> r33;

    r << r11, r12, r13,
         r21, r22, r23,
         r31, r32, r33;

    // cout << r << endl;

    q = Eigen::Quaterniond(r).normalized();
    qinv = q.inverse();

    getline(*stream,line);
    std::stringstream translation(line);
    translation >> t1;
    translation >> t2;
    translation >> t3;

    t << t1, t2, t3;

    nt = (-1) * t;

    getline(*stream, line);
    std::stringstream intrinsics(line);
    intrinsics >> fx >> chunk >> cx >> chunk >> fy >> cy >> chunk >> chunk >> chunk; 
    k << fx, 0, cx, 0, fy, cy, 0, 0, 1;
    // k << fx, 0, 0, 0, fy, 0, cx, cy, 1;
    fx = 586;
    fy = 585;
    cx = 319.5;
    cy = 239.5;

  }
};

void c2w(int counter, float scale) 
{

    // default configs
    Pose depthPose;
    depthPose.LoadStatic(576.35, 566.39, 319.5, 239.5);
    // Mat R = (Mat_<double>(3,3) << 9.9919893738676346e-01, 2.2625526733203394e-02, 3.3008621074014713e-02,
    // -2.3083319648461775e-02, 9.9964165616165557e-01, 1.3554321096587192e-02, 
    // -3.2690118983719346e-02, -1.4305411788115311e-02, 9.9936315287006794e-01);

    // Mat R = (Mat_<double>(3,3) << 9.9951523328930358e-01, 2.7517778109040710e-02,
    // -1.4562634053307221e-02, -2.7436447471273117e-02,
    // 9.9960698029782480e-01, 5.7555442853042520e-03, 1.4715290441748480e-02,
    // -5.3532072447858153e-03, 9.9987739418361155e-01);

    Mat R = (Mat_<double>(3,3) << 9.9975255232083715e-01, 9.0132979958684864e-04,
    -2.2226599662658691e-02, -4.0507407642642742e-04,
    9.9975076194767798e-01, 2.2321512046230129e-02, 2.2241178992228133e-02,
    -2.2306985220548432e-02, 9.9950374104722883e-01);

    // Mat T = (Mat_<double>(3,1) << -2.1925775089350014e-01, 2.4352843882228381e+00, 1.5679706895196575e+00);
    // Mat T = (Mat_<double>(3,1) << 1.4946831115815495e+00, 3.2909736104848988e+00, 4.2229246607019340e+00);
    Mat T = (Mat_<double>(3,1) << 2.3162195099387515e+00, 3.4346906412188671e-01, 4.5365683622308088e+00);

    // var init
    ofstream associate;

    std::vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(1);

    std::string depthname; std::string rgbname;
    std::fstream depthInfo, colorInfo;
    associate.open ("/home/worxli/Datasets/data/associate.txt");

    Mat depth;
    Size imgSize = Size(640,480);

        while(true)
        {

          // var defs
          std::stringstream depthImg, colorImg, colorParams, calibPath;
          std::ifstream depthParams;
          Pose colorPose;

          // var paths
          calibPath << "/home/worxli/Datasets/data/dense/calib" + to_string(counter) + ".txt";
          depthImg << "/home/worxli/Datasets/data/dense/depth" << counter << ".xml";
          colorImg << "/home/worxli/Datasets/data/dense/color" << counter << ".png";

          // read depth
          std::cout << "reading image " << depthImg.str() << " from xml" << std::endl;
          FileStorage fs(depthImg.str().c_str(), FileStorage::READ);
          fs["depth"] >> depth;

          // read color
          std::cout << "reading image " << colorImg.str() << std::endl;
          cv::Mat color(cv::imread(colorImg.str().c_str(), CV_LOAD_IMAGE_UNCHANGED));

          // stop if empty matrices
          if (depth.rows == 0 || color.rows == 0)
            break;

          cvtColor(color, color, CV_BGR2RGB);

          // read calib
          depthParams.open(calibPath.str().c_str());
          colorPose.Load(&depthParams);     

            // camera pose
            Eigen::Vector3d pos = colorPose.qinv.toRotationMatrix() * (-1) * colorPose.t * scale;

            // write associate file
            associate << pos[0] << " "
                      << pos[1] << " "
                      << pos[2] << " " 

                      << colorPose.qinv.x()  << " " 
                      << colorPose.qinv.y()  << " " 
                      << colorPose.qinv.z()  << " " 
                      << colorPose.qinv.w()  << " "
                      
                      << "dense/mapped" << counter << ".png "
                      << "dense/colorrgb" << counter << ".png\n";

            // new empty mat
            Mat mapped = Mat::zeros(imgSize, CV_16UC1);

            // orig pmat pointer
            uint16_t* ptrOrig = (uint16_t*)(depth.data);

            // pointer to new elems
            uint16_t* ptrMapped = (uint16_t*)(mapped.data);

            for (int row = 0; row < depth.rows; ++row)
            {
              for (int col = 0; col < depth.cols; ++col)
              {    
            
                // get depth for remapping
                double depthVal = (double)ptrOrig[depth.cols*row + col]/5;

                if(depthVal==0)
                  continue;
            
                // Map depthcam depth to 3D point
                Mat_<double> P3D = Mat_<double>(3,1);

                P3D(0) = (col - depthPose.cx) * depthVal / depthPose.fx;
                P3D(1) = (row - depthPose.cy) * depthVal / depthPose.fy;
                P3D(2) = depthVal;
            
                // Rotate and translate 3D point
                Mat_<double> P3Dp;
                P3Dp = (R*P3D) + T;
                // P3Dp = P3D;
            
                // Project 3D point to rgbcam
                double xrgb = (P3Dp(0) * colorPose.fx / P3Dp(2)) + colorPose.cx;
                double yrgb = (P3Dp(1) * colorPose.fy / P3Dp(2)) + colorPose.cy;
                double nDepth = P3Dp(2);  

                // "Interpolate" pixel coordinates (Nearest Neighbors)
                int px_rgbcam = cvRound(xrgb);
                int py_rgbcam = cvRound(yrgb);

                // cout << px_rgbcam << endl;
                // cout << py_rgbcam << endl;

                // Handle 3D occlusions
                uint16_t &depth_rgbcam = ptrMapped[depth.cols * py_rgbcam + px_rgbcam];
                // cout << nDepth << endl;

                if(px_rgbcam > 0 && py_rgbcam > 0 && px_rgbcam < depth.cols && py_rgbcam < depth.rows)
                {

                  if(depth_rgbcam == 0 || (uint16_t)nDepth < depth_rgbcam) 
                  {
                    depth_rgbcam = 5*(uint16_t)nDepth;
                    // cout << nDepth << endl;
                  }   
                } 
              }
            }

            try {
              imwrite("/home/worxli/Datasets/data/dense/mapped" + std::to_string (counter) + ".png", mapped, compression_params);
            } catch (runtime_error& ex) {
              fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
              return;
            }

            try {
              imwrite("/home/worxli/Datasets/data/dense/colorrgb" + std::to_string (counter) + ".png", color, compression_params);
            } catch (runtime_error& ex) {
              fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
              return;
            }

            counter++;
        }

        associate.close();
}

int main(int argc, char** argv)
{

    if ( argc == 3 ) {
        c2w(std::stof(argv[1]),std::stof(argv[2]));
    } else if ( argc == 2 ) {
        c2w(std::stof(argv[1]),1);
    } else{
        c2w(0,1);
    }

    return 0;
}