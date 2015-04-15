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

enum Pattern { CHESSBOARD, CIRCLES_GRID, ASYMMETRIC_CIRCLES_GRID };

static void calcChessboardCorners(Size boardSize, float squareSize, vector<Point3f>& corners, Pattern patternType = CHESSBOARD)
{
    corners.resize(0);

    for( int j = 0; j < boardSize.height; j++ )
        for( int k = 0; k < boardSize.width; k++ )
            corners.push_back(Point3f(
                                float(k*squareSize),
                                float(j*squareSize), 
                                0));
}

static bool runCalibration( vector<vector<Point2f> > imagePoints,
                    Size imageSize, Size boardSize, Pattern patternType,
                    float squareSize, float aspectRatio,
                    int flags, Mat& cameraMatrix, Mat& distCoeffs,
                    vector<Mat>& rvecs, vector<Mat>& tvecs,
                    vector<float>& reprojErrs,
                    double& totalAvgErr)
{
    cameraMatrix = Mat::eye(3, 3, CV_64F);
    if( flags & CV_CALIB_FIX_ASPECT_RATIO )
        cameraMatrix.at<double>(0,0) = aspectRatio;

    distCoeffs = Mat::zeros(8, 1, CV_64F);

    vector<vector<Point3f> > objectPoints(1);
    calcChessboardCorners(boardSize, squareSize, objectPoints[0], patternType);

    objectPoints.resize(imagePoints.size(),objectPoints[0]);

    double rms = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix,
                    distCoeffs, rvecs, tvecs, flags|CV_CALIB_FIX_K4|CV_CALIB_FIX_K5);

    printf("RMS error reported by calibrateCamera: %g\n", rms);

    bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);

    return ok;
}

void singleCameraCalibration(vector<string> imageListIR, vector<string> imageListRGB, 
              Size imageSize, Size boardSize, 
              Mat& camera1, Mat& camera2, 
              Mat& dist1, Mat& dist2,
              vector<vector<Point2f> > &imagePointsIR, vector<vector<Point2f> > &imagePointsRGB, 
              vector<vector<Point3f> > &objectPoints) {

  //check for equal list lenghts
  CV_Assert(imageListIR.size() == imageListRGB.size());
  int nimages = imageListRGB.size();

  bool displayCorners = true;
  const int maxScale = 2;
  float squareSize = 10.f, aspectRatio = 1.f;

  vector<string> imagelist;

  for( int i = 0; i < nimages; i++ )
  {
    imagelist.push_back(imageListIR[i]);
    imagelist.push_back(imageListRGB[i]);
  }

  vector<vector<Point2f> > imagePoints[2];
  imagePoints[0].resize(nimages);
  imagePoints[1].resize(nimages);

  int i, j, k;
    for( i = j = 0; i < nimages; i++ )
    {
        for( k = 0; k < 2; k++ )
        {
            const string& filename = imagelist[i*2+k];
            Mat img = imread(filename, 0);
            resize(img, img, imageSize);
            if(img.empty())
                break;
            if( imageSize == Size() )
                imageSize = img.size();
            else if( img.size() != imageSize )
            {
                cout << "The image " << filename << " has the size different from the first image size. Skipping the pair\n";
                break;
            }
            bool found = false;
            vector<Point2f>& corners = imagePoints[k][j];
            for( int scale = 1; scale <= maxScale; scale++ )
            {
                Mat timg;
                if( scale == 1 )
                    timg = img;
                else
                    resize(img, timg, Size(), scale, scale);
                found = findChessboardCorners(timg, boardSize, corners,
                    CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);
                if( found )
                {
                    if( scale > 1 )
                    {
                        Mat cornersMat(corners);
                        cornersMat *= 1./scale;
                    }
                    break;
                }
            }
            if( displayCorners )
            {
                cout << filename << endl;
                Mat cimg, cimg1;
                cvtColor(img, cimg, COLOR_GRAY2BGR);
                drawChessboardCorners(cimg, boardSize, corners, found);
                double sf = 640./MAX(img.rows, img.cols);
                resize(cimg, cimg1, Size(), sf, sf);
                imshow("corners", cimg1);
                char c = (char)waitKey(10);
                if( c == 27 || c == 'q' || c == 'Q' ) //Allow ESC to quit
                    exit(-1);
            }
            else
                putchar('.');
            if( !found )
                break;
            cornerSubPix(img, corners, Size(11,11), Size(-1,-1),
                         TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,
                                      30, 0.01));
        }
        if( k == 2 )
        {
            j++;
        }
    }
    cout << j << " pairs have been successfully detected.\n";
    nimages = j;
    if( nimages < 2 )
    {
        cout << "Error: too little pairs to run the calibration\n";
        return;
    }

    imagePoints[0].resize(nimages);
    imagePoints[1].resize(nimages);
    objectPoints.resize(nimages);

    for( i = 0; i < nimages; i++ )
    {
        for( j = 0; j < boardSize.height; j++ )
            for( k = 0; k < boardSize.width; k++ )
                objectPoints[i].push_back(Point3f(
                                float(k*squareSize),
                                float(j*squareSize), 
                                0));
    }

    // do single calibration here
    vector<Mat> rvecs, tvecs;
    vector<float> reprojErrs;
    double totalAvgErr = 0;
    bool ok;

    ok = runCalibration(imagePoints[0], imageSize, boardSize, CHESSBOARD, squareSize,
                   aspectRatio, 0, camera1, dist1,
                   rvecs, tvecs, reprojErrs, totalAvgErr);
    // printf("%s. avg reprojection error = %.2f\n",
    //        ok ? "IR camera calibration succeeded" : "IR calibration failed",
    //        totalAvgErr);

    ok = runCalibration(imagePoints[1], imageSize, boardSize, CHESSBOARD, squareSize,
                   aspectRatio, 0, camera2, dist2,
                   rvecs, tvecs, reprojErrs, totalAvgErr);
    // printf("%s. avg reprojection error = %.2f\n",
    //        ok ? "RGB calibration succeeded" : "RGB calibration failed",
    //        totalAvgErr);

    imagePointsIR = imagePoints[0];
    imagePointsRGB = imagePoints[1];
  
}

static bool readStringList( const string& filename, vector<string>& l )
{
    l.resize(0);
    FileStorage fs(filename, FileStorage::READ);
    if( !fs.isOpened() )
        return false;
    FileNode n = fs.getFirstTopLevelNode();
    if( n.type() != FileNode::SEQ )
        return false;
    FileNodeIterator it = n.begin(), it_end = n.end();
    for( ; it != it_end; ++it )
        l.push_back((string)*it);
    return true;
}

int main( int argc, char** argv ) {

	string irListName = "infrared.xml";
	string colorListName = "color.xml";
	string colorImage = "depth/col4.jpg";
	string depthMap = "depth/depth4.jpg";
	
	//read input or use defaults
	for( int i = 1; i < argc; i++ )
    {
        if( string(argv[i]) == "-cL" )
        {
            colorListName = argv[++i];
        }
        else if( string(argv[i]) == "-iL" )
        {
            irListName = argv[++i];
        }
        else if( string(argv[i]) == "-cI" )
        {
        	colorImage = argv[++i];
        }
        else if( string(argv[i]) == "-dM" )
        {
        	depthMap = argv[++i];
        }
        else
        {
            cout << "invalid option " << argv[i] << endl;
            return 0;
        }
    }

  //set defaults
  Size boardSize, imageSize;
  boardSize.width = 9;
  boardSize.height = 6;
  Pattern pattern = CHESSBOARD;

  //read image lists
  vector<string> imageListIR;
  vector<string> imageListRGB;
  if(!readStringList(irListName, imageListIR)) {
      cout << "File " << irListName << " does not exist or can't be opened!" << endl;
  }

  if(!readStringList(colorListName, imageListRGB)) {
      cout << "File " << colorListName << " does not exist or can't be opened!" << endl;
  }

  imageSize = imread(imageListRGB[0],0).size();

	//do single camera calibration
	//picture by picture from each list
	Mat camera1, camera2, dist1, dist2;
	vector<vector<Point2f> > imagePoints[2];
  vector<vector<Point3f> > objectPoints;
	singleCameraCalibration(imageListIR, imageListRGB, 
							imageSize, boardSize, 
							camera1, camera2, 
							dist1, dist2,
							imagePoints[0], imagePoints[1], objectPoints);

	//stereo calibration
	//with list filled by single calibration
	Mat R, T, E, F;
  // cout << "objectPoints: " << objectPoints[0] << endl;
  // cout << "IR Points: " << imagePoints[0][0] << endl;
  // cout << "RGB Points: " << imagePoints[1][0] << endl;

  // cout << camera1 << endl;
  // cout << camera2 << endl;

  double rms = stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],
                camera1, dist1,
                camera2, dist2,
                imageSize, R, T, E, F,
                TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, 1e-5),
               // CV_CALIB_FIX_INTRINSIC +
                CV_CALIB_USE_INTRINSIC_GUESS +
               // CV_CALIB_FIX_ASPECT_RATIO +
                //CV_CALIB_ZERO_TANGENT_DIST +
               // CV_CALIB_SAME_FOCAL_LENGTH +
                CV_CALIB_RATIONAL_MODEL +
                CV_CALIB_FIX_K3 + CV_CALIB_FIX_K4 + CV_CALIB_FIX_K5);	

  cout << R << T << endl;

  cout << "done with RMS error=" << rms << endl;
    
	//reconstruction
  //check error
    int i,j,k;
    int nimages = objectPoints.size();
    // double err = 0;
    // int npoints = 0;
    // vector<Vec3f> lines1;
    // vector<Vec3f> lines2;
    // for( i = 0; i < nimages; i++ )
    // {
    //     int npt = (int)imagePoints[0][i].size();
    //     Mat imgpt1, imgpt2;

    //     imgpt1 = Mat(imagePoints[0][i]);
    //     undistortPoints(imgpt1, imgpt1, camera1, dist1, Mat(), camera1);
    //     computeCorrespondEpilines(imgpt1, 1, F, lines1);

    //     imgpt2 = Mat(imagePoints[1][i]);
    //     undistortPoints(imgpt2, imgpt2, camera2, dist2, Mat(), camera2);
    //     computeCorrespondEpilines(imgpt2, 2, F, lines2);
        
    //     for( j = 0; j < npt; j++ )
    //     {
    //         double errij = fabs(imagePoints[0][i][j].x*lines2[j][0] +
    //                             imagePoints[0][i][j].y*lines2[j][1] + lines2[j][2]) +
    //                        fabs(imagePoints[1][i][j].x*lines1[j][0] +
    //                             imagePoints[1][i][j].y*lines1[j][1] + lines1[j][2]);
    //         err += errij;
    //     }
    //     npoints += npt;
    // }
    // cout << "average reprojection err = " <<  err/npoints << endl;

    Mat R1, R2, P1, P2, Q;
    Rect validRoi[2];

    R = R.inv();
    stereoRectify(camera1, dist1,
                  camera2, dist2,
                  imageSize, R, T, R1, R2, P1, P2, Q,
                  CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]);


    Mat rmap[2][2];
// HARTLEY'S METHOD
 // use intrinsic parameters of each camera, but
 // compute the rectification transformation directly
 // from the fundamental matrix
        // vector<Point2f> allimgpt[2];
        // for( k = 0; k < 2; k++ )
        // {
        //     for( i = 0; i < nimages; i++ )
        //         std::copy(imagePoints[k][i].begin(), imagePoints[k][i].end(), back_inserter(allimgpt[k]));
        // }
        // F = findFundamentalMat(Mat(allimgpt[0]), Mat(allimgpt[1]), FM_8POINT, 0, 0);
       //  Mat H1, H2;
       // // Mat F;

       //  F = findFundamentalMat(imagePoints[0], imagePoints[1], FM_8POINT, 0, 0);

       //  stereoRectifyUncalibrated(imagePoints[0], imagePoints[1], F, imageSize, H1, H2, 3);

       //  R1 = camera1.inv()*H1*camera1;
       //  R2 = camera2.inv()*H2*camera2;
       //  P1 = camera1;
       //  P2 = camera2;


        //Precompute maps for cv::remap()
    initUndistortRectifyMap(camera1, dist1, R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
    initUndistortRectifyMap(camera2, dist2, R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]); 


    bool isVerticalStereo = true;
    Mat canvas;
    double sf;
    int w, h;
    if( !isVerticalStereo )
    {
        sf = 600./MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width*sf);
        h = cvRound(imageSize.height*sf);
        canvas.create(h, w*2, CV_8UC3);
    }
    else
    {
        sf = 300./MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width*sf);
        h = cvRound(imageSize.height*sf);
        canvas.create(h*2, w, CV_8UC3);
    }

    for( i = 0; i < nimages; i++ )
    {

            Mat img = imread(imageListIR[0], 0), rimg, cimg;
            remap(img, rimg, rmap[0][0], rmap[0][1], CV_INTER_LINEAR);
            cvtColor(rimg, cimg, CV_GRAY2BGR);
            Mat canvasPart = !isVerticalStereo ? canvas(Rect(0, 0, w, h)) : canvas(Rect(0, 0, w, h));
            resize(cimg, canvasPart, canvasPart.size(), 0, 0, CV_INTER_AREA);

                Rect vroi(cvRound(validRoi[0].x*sf), cvRound(validRoi[0].y*sf),
                          cvRound(validRoi[0].width*sf), cvRound(validRoi[0].height*sf));
                rectangle(canvasPart, vroi, Scalar(0,0,255), 3, 8);

            img = imread(imageListRGB[0], 0);
            remap(img, rimg, rmap[1][0], rmap[1][1], CV_INTER_LINEAR);
            cvtColor(rimg, cimg, CV_GRAY2BGR);
            canvasPart = !isVerticalStereo ? canvas(Rect(w, 0, w, h)) : canvas(Rect(0, h, w, h));
            resize(cimg, canvasPart, canvasPart.size(), 0, 0, CV_INTER_AREA);

                Rect vroi2(cvRound(validRoi[1].x*sf), cvRound(validRoi[1].y*sf),
                          cvRound(validRoi[1].width*sf), cvRound(validRoi[1].height*sf));
                rectangle(canvasPart, vroi2, Scalar(0,0,255), 3, 8);

        if( !isVerticalStereo )
            for( j = 0; j < canvas.rows; j += 16 )
                line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
        else
            for( j = 0; j < canvas.cols; j += 16 )
                line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);
        
        imshow("rectified", canvas);
        std::cin.ignore();
    }


    imshow("rectified", canvas);
    std::cin.ignore();


  return 1;

}