/*
 * viewerfusion_main.cpp
 *
 *  Created on: Jun 23, 2013
 *      Author: steinbrf
 */


#include <auxiliary/multivector.h>

#include <opencv2/opencv.hpp>
#include <omp.h>
#include <stdio.h>
#include <math.h>
#include <string>
#include <fstream>
#include <sstream>
#include <Eigen/Geometry>
#include <auxiliary/debug.hpp>

//#include <fusionGPU/texturefusion.hpp>
//#include <fusionGPU/geometryfusion_single_soa.hpp>
//#include <fusionGPU/geometryfusion_single_aos.hpp>
//#include <fusionGPU/geometryfusion_dynamic_aos.hpp>
//#include <fusionGPU/geometryfusion_dynamic_multiscale.hpp>
//#include <fusionGPU/geometryfusion_mipmap.hpp>
#include <fusion/geometryfusion_mipmap_cpu.hpp>

#include "onlinefusionviewer.hpp"
#include <qapplication.h>

#include <tclap/CmdLine.h>
#include <fusion/mesh.hpp>



#define BOXDELTA 0.001
//#define VOLUMERADIUS 1.5
#define VOLUMERADIUS 1.4
#define USE_ORIGINAL_VOLUME 1

#include <fusion/definitions.h>


#include <deque>
#include <list>


 using namespace std;

CameraInfo kinectPoseFromEigen(std::pair<Eigen::Matrix3d,Eigen::Vector3d> pos,float fx, float fy, float cx, float cy){
	CameraInfo result;
	cv::Mat intrinsic = cv::Mat::eye(3,3,cv::DataType<double>::type);
	//Kinect Intrinsic Parameters
	intrinsic.at<double>(0,0) = fx;
	intrinsic.at<double>(1,1) = fy;
	intrinsic.at<double>(0,2) = cx;
	intrinsic.at<double>(1,2) = cy;

	cout << intrinsic << endl;

	result.setIntrinsic(intrinsic);
	Eigen::Matrix3d rotation = pos.first;
	cv::Mat rotation2 = cv::Mat::eye(3,3,cv::DataType<double>::type);
	for(int i=0;i<3;i++) for(int j=0;j<3;j++) rotation2.at<double>(i,j) = rotation(i,j);
	result.setRotation(rotation2);

	// cout << rotation2 << endl;

	Eigen::Vector3d translation = pos.second;
	cv::Mat translation2 = cv::Mat::zeros(3,1,cv::DataType<double>::type);
	for(int i=0;i<3;i++) translation2.at<double>(i,0) = translation(i);
	result.setTranslation(translation2);

	// cout << translation2 << endl;
	return result;
}


void filterimage(cv::Mat &image)
{
	cv::Mat input = image.clone();
	for(int x=1;x<image.cols-1;x++){
		for(int y=1;y<image.rows-1;y++){
			if(std::isfinite(input.at<float>(y,x))){
				float sum = 0.0f; float count = 0.0f;
				for(int dx=-1;dx<=1;dx++){
					for(int dy=-1;dy<=1;dy++){
						if(std::isfinite(input.at<float>(y+dy,x+dx)) && fabs(input.at<float>(y,x)-input.at<float>(y+dy,x+dx))<0.1f){
							sum += input.at<float>(y+dy,x+dx); count += 1.0f;
						}
					}
				}
				image.at<float>(y,x) = sum/count;
			}
		}
	}
}


#include <list>


int main(int argc, char *argv[])
{

	std::string meshname = "";
	bool noViewer = true;
	bool bufferImages = false;
	bool useColor = true;
	bool volumeColor = true;


	bool useLoopClosures = false;

	unsigned int startimage = 0;
	unsigned int endimage = 1000000;
	unsigned int imageStep = 1;

	float maxCamDistance = MAXCAMDISTANCE;
	float scale = DEFAULT_SCALE;
	float threshold = DEFAULT_SCALE;

	float imageDepthScale = DEPTHSCALE;

	std::string intrinsics = "";

	bool threadMeshing = true;
	bool threadFusion = false;
	bool threadImageReading = false;
	bool performIncrementalMeshing = true;

	int depthConstistencyChecks = 0;

  TCLAP::CmdLine cmdLine("onlinefusion");

  TCLAP::ValueArg<std::string> loadMeshArg("l","loadmesh","Loads this mesh",false,meshname,"string");
  TCLAP::SwitchArg threadMeshingArg("","thread-meshing","Thread the Meshing inside the Fusion",threadMeshing);
  TCLAP::SwitchArg threadFusionArg("","thread-fusion","Thread the Fusion inside the Viewer",threadFusion);
  TCLAP::SwitchArg threadImageReadingArg("","thread-image","Thread reading the Images from Hard Disk",threadImageReading);
  TCLAP::SwitchArg viewerArg("v","viewer","Show a Viewer after Fusion",!noViewer);
  TCLAP::SwitchArg bufferArg("b","buffer","Buffer all Images",bufferImages);
  TCLAP::SwitchArg useLoopClosuresArg("c","loopclosures","Read Multiple Trajectories and perform Loop Closures",useLoopClosures);
  TCLAP::SwitchArg performIncrementalMeshingArg("","incremental-meshing","Perform incremental Meshing",performIncrementalMeshing);
  TCLAP::ValueArg<int> startimageArg("s","startimage","Number of the Start Image",false,startimage,"int");
  TCLAP::ValueArg<int> endimageArg("e","endimage","Number of the End Image",false,endimage,"int");
  TCLAP::ValueArg<int> imageStepArg("k","imagestep","Use every kth step",false,imageStep,"int");
  TCLAP::ValueArg<int> depthConsistencyChecksArg("","consistency-checks","Number of Depth Consistency Checks",false,depthConstistencyChecks,"int");
  TCLAP::ValueArg<float> maxCamDistanceArg("","max-camera-distance","Maximum Camera Distance to Surface",false,maxCamDistance,"float");
  TCLAP::ValueArg<float> scaleArg("","scale","Size of the Voxel",false,scale,"float");
  TCLAP::ValueArg<float> thresholdArg("","threshold","Threshold",false,threshold,"float");
  TCLAP::ValueArg<float> imageDepthScaleArg("","imagescale","Image Depth Scale",false,imageDepthScale,"float");
  TCLAP::ValueArg<std::string> intrinsicsArg("","intrinsics","File with Camera Matrix",false,intrinsics,"string");

  TCLAP::UnlabeledMultiArg<std::string> associationfilenamesArg("filenames", "The File Names",false,"string");


  cmdLine.add(loadMeshArg);
  cmdLine.add(threadMeshingArg);
  cmdLine.add(threadFusionArg);
  cmdLine.add(threadImageReadingArg);
  cmdLine.add(viewerArg);
  cmdLine.add(bufferArg);
  cmdLine.add(useLoopClosuresArg);
  cmdLine.add(performIncrementalMeshingArg);
  cmdLine.add(startimageArg);
  cmdLine.add(endimageArg);
  cmdLine.add(imageStepArg);
  cmdLine.add(depthConsistencyChecksArg);
  cmdLine.add(maxCamDistanceArg);
  cmdLine.add(scaleArg);
  cmdLine.add(thresholdArg);
  cmdLine.add(imageDepthScaleArg);
  cmdLine.add(intrinsicsArg);

  cmdLine.add(associationfilenamesArg);
  cmdLine.parse(argc,argv);

  meshname = loadMeshArg.getValue();
  threadMeshing = threadMeshingArg.getValue();
  threadFusion = threadFusionArg.getValue();
  threadImageReading = threadImageReadingArg.getValue();
  noViewer = !viewerArg.getValue();
  bufferImages = bufferArg.getValue();
  useLoopClosures = useLoopClosuresArg.getValue();
  performIncrementalMeshing = performIncrementalMeshingArg.getValue();
  startimage = startimageArg.getValue();
  endimage = endimageArg.getValue();
  imageStep = imageStepArg.getValue();
  if(imageStep < 1) imageStep = 1;
  depthConstistencyChecks = depthConsistencyChecksArg.getValue();
  maxCamDistance = maxCamDistanceArg.getValue();
  scale = scaleArg.getValue();
  threshold = thresholdArg.getValue();
  imageDepthScale = imageDepthScaleArg.getValue();
  intrinsics = intrinsicsArg.getValue();

//lukas intriniscs
  float fx = 586.74f;
  float fy = 585.15f;
  float cx = 319.5f;
  float cy = 239.5f;
  // 584.65, 0, 319.5, 0, 580.98, 239.5

 //sample data intrinics
	// float fx = 798.445f;
 //  	float fy = 799.606f;
 //  	float cx = 476.676f;
 //  	float cy = 344.682f;


  if (intrinsics != "") {
  	fprintf(stderr,"\nReading intrinsic camera parameters from %s",intrinsics.c_str());
  	std::fstream intrinsicsfile;
  	intrinsicsfile.open(intrinsics.c_str(),std::ios::in);
  	if(!intrinsicsfile.is_open()){
  		fprintf(stderr,"\nERROR: Could not open File %s for reading!",intrinsics.c_str());
  	} else {
  		float temp;
  		fx = fy = cx = cy = -1.0f;
  		intrinsicsfile >> fx;
  		intrinsicsfile >> temp;
  		intrinsicsfile >> cx;
  		intrinsicsfile >> temp;
  		intrinsicsfile >> fy;
  		intrinsicsfile >> cy;
  		intrinsicsfile.close();
  		fprintf(stderr,"\nCamera Intrinsics from File: %f %f %f %f",fx,fy,cx,cy);
  	}
  } 

  if(threadMeshing) fprintf(stderr,"\nMeshing will run in a separate Thread");
  else              fprintf(stderr,"\nMeshing will NOT run in a separate Thread");
  if(threadFusion)  fprintf(stderr,"\nFusion will run in a separate Thread");
  if(threadImageReading) fprintf(stderr,"\nImage Reading will run in a separate Thread");



	CameraInfo startpos;

	std::string defaultname = "../test/rgbd_dataset_freiburg3_long_office_household/associationVICOM.txt";

	std::string tempfileprefix = "debug/";

	std::vector<std::string> associationfilenames = associationfilenamesArg.getValue();
	std::vector<std::string> prefices;

	if(!associationfilenames.size()){
		associationfilenames.push_back(defaultname);
	}


	for(unsigned int i=0;i<associationfilenames.size();i++){
		prefices.push_back(associationfilenames[i].substr(0,associationfilenames[i].find_last_of('/')+1));
	}



//	std::vector<Sophus::SE3> posesSophus;
	std::vector<std::pair<Eigen::Matrix3d,Eigen::Vector3d> > poses_from_assfile;
	std::vector<std::vector<std::string> > depthNames;
	std::vector<std::vector<std::string> > rgbNames;
	std::vector<std::vector<CameraInfo> > poses;

	if(!useLoopClosures){
		fprintf(stderr,"\nBuilding a single Trajectory...");
		poses.push_back(std::vector<CameraInfo>());
		depthNames.push_back(std::vector<std::string>());
		rgbNames.push_back(std::vector<std::string>());
		std::vector<CameraInfo> &trajectory = poses.back();
		std::vector<std::string> &depthNamesLast = depthNames.back();
		std::vector<std::string> &rgbNamesLast = rgbNames.back();
		unsigned int assfilestartindex = 0;
		for(unsigned int f=0;f<associationfilenames.size();f++){
			std::string prefix = prefices[f];
			std::fstream associationfile;
			std::string depthname; std::string rgbname;
			float q1, q2, q3, q4, translation1, translation2, translation3;
			associationfile.open(associationfilenames[f].c_str(),std::ios::in);
			if(!associationfile.is_open()){
				fprintf(stderr,"\nERROR: Could not open File %s",associationfilenames[f].c_str());
			}else{
				fprintf(stderr,"\nReading Association File");
				while(!associationfile.eof()){
					std::string temp("");
					getline(associationfile,temp);
					std::stringstream stream(temp);
					stream >> translation1; stream >> translation2; stream >> translation3;
					// stream >> q1; 
					stream >> q2; stream >> q3; stream >> q4; 
					stream >> q1; 
					stream >> depthname;
					cout << depthname << endl;
					if(temp!=""){
//						posesSophus.push_back(Sophus::SE3(Eigen::Quaterniond(q4,q1,q2,q3),Eigen::Vector3d(translation1,translation2,translation3)));
						poses_from_assfile.push_back(std::pair<Eigen::Matrix3d,Eigen::Vector3d>(Eigen::Quaterniond(q1,q2,q3,q4).toRotationMatrix(),Eigen::Vector3d(translation1,translation2,translation3)));
						depthNamesLast.push_back(depthname);
						if(useColor){
							stream >> rgbname; rgbNamesLast.push_back(rgbname);
						}
					}
				}
				fprintf(stderr,"\nRead %i Poses from Association File Nr %i : %s .",
						(int)depthNamesLast.size()-assfilestartindex,f,associationfilenames[f].c_str());
			}

			for(unsigned int i=assfilestartindex;i<poses_from_assfile.size();i++){
//				trajectory.push_back(kinectPoseFromSophus(posesSophus[i]));
//				trajectory.push_back(kinectPoseFromSophus(posesSophus[i],fx,fy,cx,cy));
				trajectory.push_back(kinectPoseFromEigen(poses_from_assfile[i],fx,fy,cx,cy));
				trajectory.back().setExtrinsic(startpos.getExtrinsic()*trajectory.back().getExtrinsic());
				depthNamesLast[i] = prefix + depthNamesLast[i];
				if(useColor) rgbNamesLast[i] = prefix + rgbNamesLast[i];
			}
			startpos = trajectory.back();
			assfilestartindex = depthNamesLast.size();
//			posesSophus.clear();
			poses_from_assfile.clear();
		}
	}

	if(!useLoopClosures){
		fprintf(stderr,"\nRead %i Poses und Depth Images and %i RGB Images from %i Association Files",
				(int)depthNames.size(), (int)rgbNames.size(), (int)associationfilenames.size());
	}


	if(startimage >= depthNames.front().size()) startimage = depthNames.front().size()-1;
	if(endimage >= depthNames.back().size()) endimage = depthNames.back().size()-1;


	fprintf(stderr,"\nCreating Mipmapping CPU Octree");
	FusionMipMapCPU *fusion = new FusionMipMapCPU(0,0,0,scale,threshold,0,volumeColor);

	fusion->setThreadMeshing(threadMeshing);
	fusion->setDepthChecks(depthConstistencyChecks);
	fusion->setIncrementalMeshing(performIncrementalMeshing);

	fprintf(stderr,"\nCreating Viewer");
	QApplication application(argc,argv);

	OnlineFusionViewerManipulated *viewerpointer = new OnlineFusionViewerManipulated(false);
	OnlineFusionViewerManipulated &viewer = *viewerpointer;

	fprintf(stderr,"\nSetting Viewer Parameters");
	viewer._fusion = fusion;
	viewer.setWindowTitle("Fusion Volume");
	viewer._poses = poses;
	viewer._depthNames = depthNames;
	viewer._rgbNames = rgbNames;
	viewer._threadFusion = threadFusion;
	viewer._threadImageReading = threadImageReading;
	viewer.show();
	viewer._imageDepthScale = imageDepthScale;
	viewer._maxCamDistance = maxCamDistance;
	viewer._firstFrame = (long int)startimage;
	viewer._currentFrame = (long int)startimage-1;
	fprintf(stderr,"\nSet Viewer Frame to %li",(long int)viewer._currentFrame);
	viewer._nextStopFrame = endimage;
	fprintf(stderr,"\nStarting Qt-Application");
	application.exec();

	fprintf(stderr,"\nDeleting Viewer");
	delete viewerpointer;

	fprintf(stderr,"\nProgram %s exiting.\n\n",argv[0]);
	fprintf(stderr,"\nPress Enter exit the Program");
	char input[256];
	fprintf(stderr,"%s",fgets(input,256,stdin));
	return 0;
}

