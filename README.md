# Bachelor thesis: 
This is a collection of all the files I created or modified for my bachelor thesis.

##iOS

- iPad application, called calibration
- opencv2 and structure frameworks for app

##reprojection

- reader.cpp -> converts depth data in xml files and color files as png in xmls/ to RGBD pairs in data/, calib.xml is read with the intrinsic and extrinsic parameters

##svo

- pipelineBA.cpp -> modified SVO starting point, reads png files in data/img/ color1.png to colorx.png. a starting number can be specified. saves the "associate_unscaled.txt" file with tx, ty, tz, qx, qy, qz, qw, depth file, rgb file on each line.  
- src/reprojector.cpp -> modified to export depthunscaled.txt which is a parameter for "unscale".

##conversion

- convBundle.cpp -> convert bundle.out files to associte.txt files (no scaling)
- getcloudfromdepth.cpp -> reads "associate_unscaled.txt" and produces a ply file for each RGBD pair in (X,Y,Z) coordinates and a pointcloud.obj file with all depth maps combined. Takes scaling factor as an input
- getcloudfromdepthOrigData.cpp -> same as getcloudfromdepth for example data with prescaled camera poses
- unscale.cpp -> from "associcate_unscaled.txt" and "depthunscaled.txt" produces the scaled and for fastfusion arranged "associate.txt". if no parameter is specified, scaling factor is 1
- 
##fastfusion

- onlinefusionviewer_main.cpp -> modified fastfusion main file to read our "associate.txt"
