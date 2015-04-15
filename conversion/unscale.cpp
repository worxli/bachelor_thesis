#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <Eigen/Geometry>

using namespace std;
using namespace cv;

float getScale() 
{

    std::string label;
    int line, x, y;
    float depth, realDepth;
    float scale = 1;

    double sum1 = 0, sum2 = 0;

    std::fstream associationfile;
    associationfile.open("/home/worxli/Datasets/data/depthunscaled.txt",std::ios::in);

    if(!associationfile.is_open())
    {
        fprintf(stderr,"\nERROR: Could not open File ~/Datasets/data/depthunscaled.txt\n");
    } else {
        fprintf(stderr,"\nReading depth File\n");

        while(!associationfile.eof())
        {
            std::string temp("");

            getline(associationfile,temp);

            std::stringstream stream(temp);

            stream >> label;
            stream >> line; 

            std::stringstream name;
            name << "/home/worxli/Datasets/data/depth/mapped" << line << ".png";
            // name << "/home/worxli/data/test/depth/mapped" << line << ".png";
            Mat img = cv::imread(name.str(), 0);

            cout << name.str() << endl;

            while(stream >> x != NULL) 
            {

                stream >> y;
                stream >> depth;

                uint16_t realDepth = img.at<uint16_t>(y, x);

                if(realDepth>100) {
                    sum1 = sum1 + (double) ((double) realDepth/5000 * depth);
                    sum2 = sum2 + (double) depth * depth;
                    // sum1 = sum1 + 5000.0f * depth/realDepth;
                    // sum2 = sum2 + 1; 
                }

                // outfile2 << x << " " << y << " " << realDepth << "\n" << endl;
                // outfile2 << x << " " << y << " " << depth << "\n" << endl;

                // cout << "line: " << line << " x " << x << " y " << y << " depth " << depth << " realDepth " << realDepth << endl;
            }

        }

        scale = sum1/sum2;

        // outfile2.close();

        // cout << sum1 << endl;
        // cout << sum2 << endl;
        cout << "scale " << scale << " to meters: " << scale <<  endl;
  }

  return scale;
}

void unscaleAssociate(float scale) 
{
    std::fstream associationfile;
    associationfile.open("/home/worxli/Datasets/data/associate_unscaled.txt",std::ios::in);
    // associationfile.open("/home/worxli/data/test/associate_unscaled.txt",std::ios::in);

    ofstream outfile, outfile2;
    outfile.open ("/home/worxli/Datasets/data/associate.txt");
    outfile2.open ("/home/worxli/Datasets/data/svo_cameras.ply");

    // outfile.open ("/home/worxli/data/test/associate.txt");
    // outfile2.open ("/home/worxli/data/test/svo_cameras.ply");

    outfile2 << "ply\n"
          << "format ascii 1.0\n"
          << "element face 0\n"
          << "property list uchar int vertex_indices\n"
          << "element vertex 88\n"
          << "property float x\n"
          << "property float y\n"
          << "property float z\n"
          << "end_header\n";

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

            stream >> translation1; stream >> translation2; stream >> translation3;
            stream >> q2; stream >> q3; stream >> q4; stream >> q1; 
            stream >> depthname;
            stream >> rgbname;

            Eigen::Matrix3d r = Eigen::Quaterniond(q1,q2,q3,q4).toRotationMatrix();
            Eigen::Vector3d t = Eigen::Vector3d(translation1 * scale, translation2 * scale, translation3 * scale);

            Eigen::Vector3d vec = ((-1) * r.transpose()) * t;
            // Eigen::Vector3d vec = ((-1) * r.transpose()) * t;

            // outfile << vec[0] << " "
            //     << vec[1] << " "
            //     << vec[2] << " "


            // t = (-1)*t;

            Eigen::Quaterniond qinv = Eigen::Quaterniond(q1,q2,q3,q4).inverse();
            Eigen::Vector3d pos = qinv.toRotationMatrix() * (-1) * t;

            outfile << pos[0] << " "
                << pos[1] << " "
                << pos[2] << " " 
                << qinv.x() << " " 
                << qinv.y() << " " 
                << qinv.z() << " " 
                << qinv.w() << " "  
                << depthname << " "
                << rgbname << "\n";

            outfile2 << vec[0]  << " " << vec[1] << " " << vec[2] << endl;


        }

        outfile.close();
        outfile2.close();
        cout << "wrote associate file" << endl;
  }
}

int main(int argc, char** argv)
{

    if ( argc != 2 ) {
        float scale = getScale();
        unscaleAssociate(scale);
    } else {
        unscaleAssociate(std::stof(argv[1]));
    }

    return 0;
}