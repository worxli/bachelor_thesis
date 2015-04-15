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

  }
};

void c2w(int counter) 
{

    ofstream outfile, outfile2, cameras, associate;

    std::vector<Eigen::Vector3d> pointcloud;
    std::vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(1);

    std::string depthname; std::string rgbname;
    std::fstream depthInfo, colorInfo;
    cameras.open ("/home/worxli/Datasets/data/cameras.ply");
    associate.open ("/home/worxli/Datasets/data/associate_unscaled.txt");

    cameras << "ply\n"
                  << "format ascii 1.0\n"
                  << "element face 0\n"
                  << "property list uchar int vertex_indices\n"
                  << "element vertex 30\n"
                  << "property float x\n"
                  << "property float y\n"
                  << "property float z\n"
                  << "end_header\n";

        while(true)
        {

          std::stringstream depthImg, colorImg, colorParams;
          depthImg << "/home/worxli/Datasets/data/output/depth" << counter << ".png";
          std::cout << "reading image " << depthImg.str() << std::endl;
          cv::Mat depth(cv::imread(depthImg.str().c_str(), CV_LOAD_IMAGE_UNCHANGED));
          depth = depth*5;

          colorImg << "/home/worxli/Datasets/data/dense/color" << counter << ".png";
          std::cout << "reading image " << colorImg.str() << std::endl;
          cv::Mat image(cv::imread(colorImg.str().c_str(), CV_LOAD_IMAGE_UNCHANGED));

          try {
            imwrite("/home/worxli/Datasets/data/output/mapped" + to_string(counter) + ".png", depth, compression_params);
          } catch(runtime_error& er) {
            cout << "error" << endl;
          }

          if (depth.rows==0)
            break;

          std::ifstream depthParams;
          depthParams.open("/home/worxli/Datasets/data/dense/calib" + to_string(counter) + ".txt");
          Pose depthPose;
          depthPose.Load(&depthParams);

            std::stringstream fname, fname2;
            fname2 << "/home/worxli/Datasets/data/rgb" << counter << ".ply";
            outfile2.open (fname2.str());

            outfile2 << "ply\n"
                  << "format ascii 1.0\n"
                  << "element face 0\n"
                  << "property list uchar int vertex_indices\n"
                  << "element vertex 7000\n"
                  << "property float x\n"
                  << "property float y\n"
                  << "property float z\n"
                  << "property uchar red\n"
                  << "property uchar green\n"
                  << "property uchar blue\n"
                  << "end_header\n";            

            // write camera pose
            Eigen::Vector3d pos = ((-1) * depthPose.r.transpose()) * depthPose.t;
            cameras << pos[0] << " " << pos[1] << " " << pos[2] << endl;

            // write associate file
            associate << depthPose.t(0) << " "
                      << depthPose.t(1) << " "
                      << depthPose.t(2) << " " 

                      << depthPose.q.x()  << " " 
                      << depthPose.q.y()  << " " 
                      << depthPose.q.z()  << " " 
                      << depthPose.q.w()  << " "
                      
                      << "output/mapped" << counter << ".png "
                      << "dense/color" << counter << ".png\n";

            for (int y = 0; y < depth.rows; y=y+4)
             {
                for (int x = 0; x < depth.cols; x=x+4)
                 {
                    uint16_t realDepth = depth.at<uint16_t>(y, x);
                    double realDepth2 = realDepth * 0.0002f;

                    if(realDepth>0)
                    {

                        Eigen::Vector3d u = Eigen::Vector3d((double)x*realDepth2, (double)y*realDepth2, realDepth2);
                        Eigen::Vector3d vec = depthPose.qinv.toRotationMatrix() * ( depthPose.k.inverse() * u - depthPose.t);

                        pointcloud.push_back(vec);

                        Vec3b color = image.at<Vec3b>(y, x);
                        outfile2 << vec[0] << " " << vec[1] << " " << vec[2];
                        outfile2 << " " << (int)color.val[2] << " " << (int)color.val[1] << " " << (int)color.val[0] << endl;
                    }
                }
            }

            outfile2.close();

            counter++;
        }
        cout << "wrote ply depth file" << endl;
        cameras.close();

        std::ofstream pointcloud_file("/home/worxli/Datasets/data/pointcloud.obj");
        pointcloud_file << "g pointcloud" << std::endl;
        for (Eigen::Vector3d point : pointcloud) {
          pointcloud_file << "v " << point(0) << " " << point(1) << " " << point(2) << std::endl;
        }
        pointcloud_file.close();
}

int main(int argc, char** argv)
{

    if ( argc == 2 ) {
        c2w(std::stof(argv[1]));
    } else {
        c2w(0);
    }

    return 0;
}