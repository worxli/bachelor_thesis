#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <Eigen/Geometry>

using namespace std;
using namespace cv;

void c2w(float transscale) 
{

    Eigen::Matrix3d k;
    k << 586.74, 0, 319.5, 0, 585.15, 239.5, 0, 0, 1;
    // k << 525, 0, 319.5, 0, 525, 239.5, 0, 0, 1;
    // 575.8, 568.8, 319.5, 239.5);

    // double fx = 575.8;
    // double fy = 568.8;
    // double cx = 319.5;
    // double cy = 239;

    int counter = 0;
    std::vector<Eigen::Vector3d> pointcloud;
    std::fstream associationfile;
    associationfile.open("/home/worxli/Datasets/data/associate_unscaled.txt",std::ios::in);

    ofstream outfile, outfile2;

    if(!associationfile.is_open())
    {
        fprintf(stderr,"\nERROR: Could not open File\n");
    } else {
        fprintf(stderr,"\nReading Association unscaled File\n");

        std::string depthname; std::string rgbname;
        float q1, q2, q3, q4, translation1, translation2, translation3;

        while(!associationfile.eof())
        {
            std::string temp("");
            getline(associationfile,temp);            

            std::stringstream stream(temp);

            std::stringstream fname, fname2;
            fname2 << "/home/worxli/Datasets/data/rgb" << counter << ".ply";
            outfile2.open(fname2.str());

            outfile2 << "ply\n"
                  << "format ascii 1.0\n"
                  << "element face 0\n"
                  << "property list uchar int vertex_indices\n"
                  << "element vertex 10000\n"
                  << "property float x\n"
                  << "property float y\n"
                  << "property float z\n"
                  << "property uchar red\n"
                  << "property uchar green\n"
                  << "property uchar blue\n"
                  << "end_header\n";

            stream >> translation1; stream >> translation2; stream >> translation3;
            stream >> q2; stream >> q3; stream >> q4; stream >> q1; 

            stream >> depthname;
            stream >> rgbname;

            std::stringstream name, name2;
            name << "/home/worxli/Datasets/data/" << depthname;
            Mat img = cv::imread(name.str(), -1);
            name2 << "/home/worxli/Datasets/data/" << rgbname;
            Mat rgb = cv::imread(name2.str(), 1);
            
            //tx ty tz qx qy qz qw
            Eigen::Matrix3d r = Eigen::Quaterniond(q1,q2,q3,q4).toRotationMatrix();

            Eigen::Vector3d t = Eigen::Vector3d(translation1 * transscale, translation2 * transscale, translation3 * transscale);

            //with U = [D*u; D*v; D]'  // coordinate in pixels (u,v), depth D, intrinsic K, extrinsics [R T], 3D point X
            //so X = inv(R) * ( inv(K) * U - T )

            cout << "image: " << name.str() << endl;

            for (int y = 0; y < img.rows; y=y+4)
             {
                for (int x = 0; x < img.cols; x=x+4)
                 {
                    uint16_t realDepth = img.at<uint16_t>(y, x);
                    double realDepth2 = (double)realDepth*0.0002f;

                    if(realDepth>100)
                    {

                        Eigen::Vector3d u = Eigen::Vector3d((double)x*realDepth2, (double)y*realDepth2, realDepth2);
                        Eigen::Vector3d vec = r.transpose() * ( k.inverse() * u - t);

                        pointcloud.push_back(vec);

                        Vec3b color = rgb.at<Vec3b>(y, x);
                        outfile2 << vec[0] << " " << vec[1] << " " << vec[2];
                        outfile2 << " " << (int)color.val[2] << " " << (int)color.val[1] << " " << (int)color.val[0] << endl;
                    }
                }
            }

            outfile2.close();
            counter++;
        }
        cout << "wrote ply depth file" << endl;

        std::ofstream pointcloud_file("/home/worxli/Datasets/data/pointcloud.obj");
        pointcloud_file << "g pointcloud" << std::endl;
        for (Eigen::Vector3d point : pointcloud) {
          pointcloud_file << "v " << point(0) << " " << point(1) << " " << point(2) << std::endl;
        }
        pointcloud_file.close();
  }
}

int main(int argc, char** argv)
{

    if ( argc == 2 ) {
        c2w(std::stof(argv[1]));
    } else {
        c2w(1);
    }

    return 0;
}