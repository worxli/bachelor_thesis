#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

void rewrite() 
{

    std::string rgb, depth;
    float chunk;
    int i = 0;

    std::fstream associationfile;
    associationfile.open("/home/worxli/data/test_orig/associate.txt",std::ios::in);

    if(!associationfile.is_open())
    {
        fprintf(stderr,"\nERROR: Could not open File\n");
    } else {
        fprintf(stderr,"\nReading associate File\n");

        while(!associationfile.eof())
        {
            std::string temp("");
            getline(associationfile,temp);
            std::stringstream stream(temp);

            stream >> chunk; stream >> chunk; stream >> chunk;
            stream >> chunk; stream >> chunk; stream >> chunk;
            stream >> chunk; stream >> chunk; stream >> chunk;
            stream >> depth;
            stream >> chunk;
            stream >> rgb;

            std::stringstream depthname, rgbname, depthnew, rgbnew;
            depthname << "/home/worxli/data/test_orig/depth/" << depth.substr(6);
            rgbname << "/home/worxli/data/test_orig/rgb/" << rgb.substr(4);
            depthnew << "/home/worxli/data/test/depth/mapped" << i << ".png";
            rgbnew << "/home/worxli/data/test/img/color" << i << ".png";

            cout << depthname.str() << endl;
            cout << rgbname.str() << endl;

            std::ifstream  src(depthname.str(), std::ios::binary);
            std::ofstream  dst(depthnew.str(), std::ios::binary);

            dst << src.rdbuf();

            std::ifstream  src2(rgbname.str(), std::ios::binary);
            std::ofstream  dst2(rgbnew.str(), std::ios::binary);

            dst2 << src2.rdbuf();
            i++;

        }
  }
}


int main(int argc, char** argv)
{

    rewrite();

    return 0;
}