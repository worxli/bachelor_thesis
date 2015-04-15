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

enum { DETECTION = 0, CAPTURING = 1, CALIBRATED = 2 };
enum Pattern { CHESSBOARD, CIRCLES_GRID, ASYMMETRIC_CIRCLES_GRID };
vector<Point3f> Create3DChessboardCorners(Size boardSize, float squareSize);

static double computeReprojectionErrors(
        const vector<vector<Point3f> >& objectPoints,
        const vector<vector<Point2f> >& imagePoints,
        const vector<Mat>& rvecs, const vector<Mat>& tvecs,
        const Mat& cameraMatrix, const Mat& distCoeffs,
        vector<float>& perViewErrors )
{
    vector<Point2f> imagePoints2;
    int i, totalPoints = 0;
    double totalErr = 0, err;
    perViewErrors.resize(objectPoints.size());

    for( i = 0; i < (int)objectPoints.size(); i++ )
    {
        projectPoints(Mat(objectPoints[i]), rvecs[i], tvecs[i],
                      cameraMatrix, distCoeffs, imagePoints2);
        err = norm(Mat(imagePoints[i]), Mat(imagePoints2), CV_L2);
        int n = (int)objectPoints[i].size();
        perViewErrors[i] = (float)sqrt(err*err/n);
        totalErr += err*err;
        totalPoints += n;
    }

    return sqrt(totalErr/totalPoints);
}

static void calcChessboardCorners(Size boardSize, float squareSize, vector<Point3f>& corners, Pattern patternType = CHESSBOARD)
{
    corners.resize(0);

    for( int i = 0; i < boardSize.height; i++ )
        for( int j = 0; j < boardSize.width; j++ )
            corners.push_back(Point3f(float(j*squareSize),
                                      float(i*squareSize), 0));
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
                    ///*|CV_CALIB_FIX_K3*/|CV_CALIB_FIX_K4|CV_CALIB_FIX_K5);
    printf("RMS error reported by calibrateCamera: %g\n", rms);

    bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);

    totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints,
                rvecs, tvecs, cameraMatrix, distCoeffs, reprojErrs);

    return ok;
}


static void saveCameraParams( const string& filename,
                       Size imageSize, Size boardSize,
                       float squareSize, float aspectRatio, int flags,
                       const Mat& cameraMatrix, const Mat& distCoeffs,
                       const vector<Mat>& rvecs, const vector<Mat>& tvecs,
                       const vector<float>& reprojErrs,
                       const vector<vector<Point2f> >& imagePoints,
                       double totalAvgErr )
{
    FileStorage fs( filename, FileStorage::WRITE );

    time_t tt;
    time( &tt );
    struct tm *t2 = localtime( &tt );
    char buf[1024];
    strftime( buf, sizeof(buf)-1, "%c", t2 );

    fs << "calibration_time" << buf;

    if( !rvecs.empty() || !reprojErrs.empty() )
        fs << "nframes" << (int)max(rvecs.size(), reprojErrs.size());
    fs << "image_width" << imageSize.width;
    fs << "image_height" << imageSize.height;
    fs << "board_width" << boardSize.width;
    fs << "board_height" << boardSize.height;
    fs << "square_size" << squareSize;

    if( flags & CV_CALIB_FIX_ASPECT_RATIO )
        fs << "aspectRatio" << aspectRatio;

    if( flags != 0 )
    {
        sprintf( buf, "flags: %s%s%s%s",
            flags & CV_CALIB_USE_INTRINSIC_GUESS ? "+use_intrinsic_guess" : "",
            flags & CV_CALIB_FIX_ASPECT_RATIO ? "+fix_aspectRatio" : "",
            flags & CV_CALIB_FIX_PRINCIPAL_POINT ? "+fix_principal_point" : "",
            flags & CV_CALIB_ZERO_TANGENT_DIST ? "+zero_tangent_dist" : "" );
        cvWriteComment( *fs, buf, 0 );
    }

    fs << "flags" << flags;

    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;

    fs << "avg_reprojection_error" << totalAvgErr;
    if( !reprojErrs.empty() )
        fs << "per_view_reprojection_errors" << Mat(reprojErrs);

    if( !rvecs.empty() && !tvecs.empty() )
    {
        CV_Assert(rvecs[0].type() == tvecs[0].type());
        Mat bigmat((int)rvecs.size(), 6, rvecs[0].type());
        for( int i = 0; i < (int)rvecs.size(); i++ )
        {
            Mat r = bigmat(Range(i, i+1), Range(0,3));
            Mat t = bigmat(Range(i, i+1), Range(3,6));

            CV_Assert(rvecs[i].rows == 3 && rvecs[i].cols == 1);
            CV_Assert(tvecs[i].rows == 3 && tvecs[i].cols == 1);
            //*.t() is MatExpr (not Mat) so we can use assignment operator
            r = rvecs[i].t();
            t = tvecs[i].t();
        }
        cvWriteComment( *fs, "a set of 6-tuples (rotation vector + translation vector) for each view", 0 );
        fs << "extrinsic_parameters" << bigmat;
    }

    if( !imagePoints.empty() )
    {
        Mat imagePtMat((int)imagePoints.size(), (int)imagePoints[0].size(), CV_32FC2);
        for( int i = 0; i < (int)imagePoints.size(); i++ )
        {
            Mat r = imagePtMat.row(i).reshape(2, imagePtMat.cols);
            Mat imgpti(imagePoints[i]);
            imgpti.copyTo(r);
        }
        fs << "image_points" << imagePtMat;
    }
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

static bool runAndSave(const string& outputFilename,
                const vector<vector<Point2f> >& imagePoints,
                Size imageSize, Size boardSize, Pattern patternType, float squareSize,
                float aspectRatio, int flags, Mat& cameraMatrix,
                Mat& distCoeffs, bool writeExtrinsics, bool writePoints )
{
    vector<Mat> rvecs, tvecs;
    vector<float> reprojErrs;
    double totalAvgErr = 0;

    bool ok = runCalibration(imagePoints, imageSize, boardSize, patternType, squareSize,
                   aspectRatio, flags, cameraMatrix, distCoeffs,
                   rvecs, tvecs, reprojErrs, totalAvgErr);
    printf("%s. avg reprojection error = %.2f\n",
           ok ? "Calibration succeeded" : "Calibration failed",
           totalAvgErr);

    // if( ok )
    //     saveCameraParams( outputFilename, imageSize,
    //                      boardSize, squareSize, aspectRatio,
    //                      flags, cameraMatrix, distCoeffs,
    //                      writeExtrinsics ? rvecs : vector<Mat>(),
    //                      writeExtrinsics ? tvecs : vector<Mat>(),
    //                      writeExtrinsics ? reprojErrs : vector<float>(),
    //                      writePoints ? imagePoints : vector<vector<Point2f> >(),
    //                      totalAvgErr );
    return ok;
}

int singleCalibration(vector<string> imageList, 
    vector<vector<Point2f> > &imagePoints, 
    Size imageSize, Size boardSize, 
    int nframes, Pattern pattern, 
    float squareSize, float aspectRatio,
    Mat& cameraMatrix, Mat& distCoeffs,
    int mode) {

    int flags = 0;
    bool writeExtrinsics = false;
    bool writePoints = false;
    bool flipVertical = false;
    bool undistortImage = false;

    vector<Point2f> pointbuf;

    for(int i = 0;;i++)
    {
        Mat view, viewGray;
        bool blink = false;

        if( i < (int)imageList.size() ) {
            view = imread(imageList[i], 1);
            resize(view, view, imageSize);
        }
            

        if(!view.data)
        {
            if( imagePoints.size() > 0 ) {
                runAndSave("someFileName.xml", imagePoints, imageSize,
                           boardSize, pattern, squareSize, aspectRatio,
                           flags, cameraMatrix, distCoeffs,
                           writeExtrinsics, writePoints);             
            }

            break;
        }

        cout << "filename: " << imageList[i] << endl;

        if( flipVertical )
            flip( view, view, 0 );

        cvtColor(view, viewGray, COLOR_BGR2GRAY);

        bool found = findChessboardCorners( view, boardSize, pointbuf,
                    CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);

       // improve the found corners' coordinate accuracy
        if( pattern == CHESSBOARD && found) cornerSubPix( viewGray, pointbuf, Size(11,11),
            Size(-1,-1), TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));

        imagePoints.push_back(pointbuf);


        if(found) {
            drawChessboardCorners( view, boardSize, Mat(pointbuf), found );
        } else {
            printf("corners not found\n");
        }
            

        // string msg = mode == CAPTURING ? "100/100" :
        //     mode == CALIBRATED ? "Calibrated" : "Press 'g' to start";
        // int baseLine = 0;
        // Size textSize = getTextSize(msg, 1, 1, 1, &baseLine);
        // Point textOrigin(view.cols - 2*textSize.width - 10, view.rows - 2*baseLine - 10);

        // putText( view, msg, textOrigin, 1, 1,
        //          mode != CALIBRATED ? Scalar(0,0,255) : Scalar(0,255,0));

        // if( blink )
        //     bitwise_not(view, view);

        // if( mode == CALIBRATED && undistortImage )
        // {
        //     Mat temp = view.clone();
        //     undistort(temp, view, cameraMatrix, distCoeffs);
        // }

        imshow("Image View", view);

        // std::cin.ignore();
        // destroyWindow
        // int key = 0xff & waitKey(capture.isOpened() ? 50 : 500);

        // if( (key & 255) == 27 )
        //     break;

        // if( key == 'u' && mode == CALIBRATED )
        //     undistortImage = !undistortImage;

        // if( mode == CAPTURING && imagePoints.size() >= (unsigned)nframes )
        // {
        //     if( runAndSave(outputFilename, imagePoints, imageSize,
        //                boardSize, pattern, squareSize, aspectRatio,
        //                flags, cameraMatrix, distCoeffs,
        //                writeExtrinsics, writePoints))
        //         mode = CALIBRATED;
        //     else
        //         mode = DETECTION;
        //     if( !capture.isOpened() )
        //         break;
        // }
    }

    //cout << cameraMatrix << endl; 

    return 1;
}



int main( int argc, char** argv )
{
    Size boardSize, imageSize;
    float squareSize = 1.f, aspectRatio = 1.f;
    Mat cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2;
    const char* outputFilename = "out_camera_data.yml";

    int i, nframes;
    bool writeExtrinsics = false, writePoints = false;
    bool undistortImage = false;
    int flags = 0;
    VideoCapture capture;
    bool flipVertical = false;
    bool showUndistorted = false;
    bool videofile = false;
    int delay = 1000;
    clock_t prevTimestamp = 0;
    int mode = DETECTION;
    int cameraId = 0;
    vector<vector<Point2f> > imagePointsIR;
    vector<vector<Point2f> > imagePointsRGB;
    vector<string> imageListIR;
    vector<string> imageListRGB;
    

    // defaults
    boardSize.width = 9;
    boardSize.height = 6;
    imageSize.width = 640;
    imageSize.height = 480;

    Pattern pattern = CHESSBOARD;

    const char* inputFilenameIR = "infrared.xml";
    const char* inputFilenameRGB = "color.xml";

    if(!readStringList(inputFilenameIR, imageListIR)) {
        cout << "File " << inputFilenameIR << " does not exist or can't be opened!" << endl;
    }

    if(!readStringList(inputFilenameRGB, imageListRGB)) {
        cout << "File " << inputFilenameRGB << " does not exist or can't be opened!" << endl;
    }

    singleCalibration(imageListIR, imagePointsIR, imageSize, boardSize, (int)imageListIR.size(), pattern, squareSize, aspectRatio, 
        cameraMatrix1, distCoeffs1, mode);

    singleCalibration(imageListRGB, imagePointsRGB, imageSize, boardSize, (int)imageListRGB.size(), pattern, squareSize, aspectRatio, 
        cameraMatrix2, distCoeffs2, mode);

    Mat R, T, E, F;
    CV_Assert(imageListIR.size() == imageListRGB.size());
    nframes = imageListIR.size();
    
    vector<vector<Point3f> > objectPoints(nframes);
    for(i=0;i<nframes;i++) {
        objectPoints[i] = Create3DChessboardCorners(boardSize, squareSize);
    }

    double rms = stereoCalibrate(objectPoints, imagePointsIR, imagePointsRGB,
                  cameraMatrix1, distCoeffs1,
                  cameraMatrix2, distCoeffs2,
                  imageSize, R, T, E, F,
                  TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, 1e-5),
                  CV_CALIB_FIX_INTRINSIC +
                  CV_CALIB_USE_INTRINSIC_GUESS +
                  CV_CALIB_FIX_ASPECT_RATIO +
                  CV_CALIB_ZERO_TANGENT_DIST +
                  CV_CALIB_SAME_FOCAL_LENGTH +
                  CV_CALIB_RATIONAL_MODEL +
                  CV_CALIB_FIX_K3 + CV_CALIB_FIX_K4 + CV_CALIB_FIX_K5);

    cout << "done with RMS error=" << rms << endl;

    // double rms = stereoCalibrate(objectPoints, imagePointsRGB, imagePointsIR,
    //               cameraMatrix2, distCoeffs2,
    //               cameraMatrix1, distCoeffs1,
    //               imageSize, R, T, E, F,
    //               TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, 1e-5),
    //               CV_CALIB_FIX_INTRINSIC +
    //               CV_CALIB_USE_INTRINSIC_GUESS +
    //               CV_CALIB_FIX_ASPECT_RATIO +
    //               CV_CALIB_ZERO_TANGENT_DIST +
    //               CV_CALIB_SAME_FOCAL_LENGTH +
    //               CV_CALIB_RATIONAL_MODEL +
    //               CV_CALIB_FIX_K3 + CV_CALIB_FIX_K4 + CV_CALIB_FIX_K5);

    // cout << "done with RMS error=" << rms << endl;


    Mat ir = imread(imageListIR[0],0);
    Mat dep = imread("depth/depth4.jpg",0);
    Mat rgb = imread("depth/col4.jpg",0);
    //Mat rgbd = Mat(imageSize.width, imageSize.height, CV_64F, 4);

    //bitwise_not(dep,dep);

    imshow("depth", dep);
    std::cin.ignore();

    imshow("color", rgb);
    std::cin.ignore();

    // cout << dep << endl;
    //dep.convertTo(dep, CV_64FC1);
    // std::cin.ignore();
    // cout << dep << endl;

    imshow("depth float", dep);
    std::cin.ignore();

    // cout << R << endl;
    // cout << T << endl;

    //std::cin.ignore();

    Mat depthMap = Mat::zeros(imageSize.height,imageSize.width,CV_32FC1); 


    Mat depthmap_rgbcam = cv::Mat::zeros(imageSize.height,imageSize.width,CV_32FC1); // Initialize the depthmap to all zeros
    float *depthmap_rgbcam_buffer = (float*)depthmap_rgbcam.data;
    for( int row = 0; row < imageSize.height; row++ ) {
        for( int col = 0; col < imageSize.width; col++ ) {

            float depth = dep.at<uchar>(row,col);
            // if(depth==0.f)
            //     break;

            //cout << "depth: " << depth << endl;
            depth = -(depth-255)/255.f*9.f; //convert to mm
            //cout << "depth: " << depth << "m" << endl;
            //cout << " converted " << depth << endl; 
            // dep.at<double>(row,col) = depth;

            float depth_depthcam = depth;//disparity_to_depth(depth);
            //Point3f color = rgb.at<double>(i,j);
            
            Mat_<double> P3D(3,1);
            P3D(0) = (col - cameraMatrix1.at<double>(0,2)) * depth / cameraMatrix1.at<double>(0,0);
            P3D(1) = (row - cameraMatrix1.at<double>(1,2)) * depth / cameraMatrix1.at<double>(1,1);
            P3D(2) = depth;

            //cout << P3D << endl;

            //cout << cameraMatrix1 << endl;

            Mat_<double> P3Dp; //projected onto color img
            Point3f P2D_rgb = Point3f();

            P3Dp = (R*P3D) + T;
            float xrgb = (P3Dp(0) * cameraMatrix2.at<double>(0,0) / P3Dp(2)) + cameraMatrix2.at<double>(0,2);
            float yrgb = (P3Dp(1) * cameraMatrix2.at<double>(1,1) / P3Dp(2)) + cameraMatrix2.at<double>(1,2);

            cout << "xrgb " << xrgb << " yrgb " << yrgb << " depth " << P3Dp(2) << " change: " << P3Dp(2)-depth << endl;

            // "Interpolate" pixel coordinates (Nearest Neighbors, as discussed above)
            int px_rgbcam = cvRound(xrgb);
            int py_rgbcam = cvRound(yrgb);

            //cout << px_rgbcam << " x - y " << py_rgbcam << endl;

            //cout << "x " << col << " y " << row << endl; 

            // Handle 3D occlusions
            // float &depth_rgbcam = depthmap_rgbcam_buffer[imageSize.width*py_rgbcam+px_rgbcam];
            // if(depth_rgbcam==0 || depth_depthcam<depth_rgbcam)
            //     depth_rgbcam = depth_depthcam;

            //cout << P3Dp(2) << endl;

            if(px_rgbcam<=imageSize.width && py_rgbcam <= imageSize.height)
                depthMap.at<float>(py_rgbcam,px_rgbcam) = P3Dp(2)/9*255;

        }
    }

    cout << "map" << endl;

    imshow("depth new", dep);
    std::cin.ignore();

    imshow("depth new2", depthmap_rgbcam);
    std::cin.ignore();

    imshow("depthMap", depthMap);
    std::cin.ignore();


    // Mat R1, R2, P1, P2, Q;
    // Rect validRoi[2];

    // stereoRectify(cameraMatrix1, distCoeffs1,
    //               cameraMatrix2, distCoeffs2,
    //               imageSize, R, T, R1, R2, P1, P2, Q,
    //               CV_CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]);

    // Mat rmap[2][2];
    // int nimages = nframes;
    // int k;
    //     //Precompute maps for cv::remap()
    // initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
    // initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

    // for( i = 0; i < nimages; i++ )
    // {
    //     Mat i1, i2;
    //     Mat im1 = imread(imageListIR[i], 1);
    //     Mat im2 = imread(imageListRGB[i], 1);
    //     remap(im1, i1, rmap[0][0], rmap[0][1], CV_INTER_LINEAR);
    //     remap(im2, i2, rmap[1][0], rmap[1][1], CV_INTER_LINEAR);

    //     imshow("image1", im1);
    //     waitKey(500);

    //     imshow("image2", im2);
    //     waitKey(500);

    //     imshow("im1", i1);
    //     waitKey(500);
    //     imshow("im2", i2);
    //     waitKey(500);
    // }

    

    return 0;
}

vector<Point3f> Create3DChessboardCorners(Size boardSize, float squareSize)
{
  // This function creates the 3D points of your chessboard in its own coordinate system

  vector<Point3f> corners;

  for( int i = 0; i < boardSize.height; i++ )
  {
    for( int j = 0; j < boardSize.width; j++ )
    {
      corners.push_back(Point3f(float(j*squareSize),
                                float(i*squareSize), 0));
    }
  }

  return corners;
}
