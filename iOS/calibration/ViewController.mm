//
//  ViewController.m
//  calibration
//
//  Created by Lukas Bischofberger on 09/01/15.
//  Copyright (c) 2015 Lukas Bischofberger. All rights reserved.
//

#import "ViewController.h"

#import <AVFoundation/AVFoundation.h>

#import <Structure/StructureSLAM.h>
#import <opencv2/opencv.hpp>

#include <algorithm>

using namespace cv;

struct AppStatus
{
    NSString* const pleaseConnectSensorMessage = @"Please connect Structure Sensor.";
    NSString* const pleaseChargeSensorMessage = @"Please charge Structure Sensor.";
    NSString* const needColorCameraAccessMessage = @"This app requires camera access to capture color.\nAllow access by going to Settings → Privacy → Camera.";
    
    enum SensorStatus
    {
        SensorStatusOk,
        SensorStatusNeedsUserToConnect,
        SensorStatusNeedsUserToCharge,
    };
    
    // Structure Sensor status.
    SensorStatus sensorStatus = SensorStatusOk;
    
    // Whether iOS camera access was granted by the user.
    bool colorCameraIsAuthorized = true;
    
    // Whether there is currently a message to show.
    bool needsDisplayOfStatusMessage = false;
    
    // Flag to disable entirely status message display.
    bool statusMessageDisabled = false;
};

@interface ViewController () <AVCaptureVideoDataOutputSampleBufferDelegate> {

   	STSensorController *_sensorController;
    
    AVCaptureSession *_avCaptureSession;
    AVCaptureDevice *_videoDevice;
    bool _avCaptureSupportsFocusNearInfinity;

    UIImageView *_depthImageView;
    UIImageView *_normalsImageView;
    UIImageView *_colorImageView;
    UIImageView *_irImageView;
    
    uint16_t *_linearizeBuffer;
    uint8_t *_coloredDepthBuffer;
    uint8_t *_normalsBuffer;

    STFloatDepthFrame *_floatDepthFrame;
    STFloatDepthFrame *_lastFloatDepth;
    STNormalEstimator *_normalsEstimator;
    
    // last received frames
    UIImage *_lastColorImage;
    UIImage *_lastInfraredImage;
    //STFrame *_lastDepthFrame;

    //labels
    UILabel* _statusLabel;
    UILabel *_calibCounter;
    UILabel *_text2;
    UILabel *_text3;
    UILabel *_text4;
    UILabel *_text5;
    
    UIButton *_calib;
    UIButton *_stereoCalib;
    UIButton *_restartStream;
    UIButton *_showIR;
    UIButton *_showDepth;
    UIButton *_showBoth;
    UIButton *_showIRSync;
    UIButton *_showDepthSync;
    UIButton *_toggleDist;
    UIButton *_remapDepth;
    UIButton *_recordSeq;
    UIButton *_saveCalib;
    UIButton *_recordSeqAuto;
    
    AppStatus _appStatus;
    
    int counter;
    
    // calibration stuff
    bool streaming;
    bool makeNew;
    bool useUndist;
    bool remap3D;
    bool sequence;
    bool recording;
    
    vector<vector<Point2f> > imagePoints[2];
    vector<vector<Point3f> > objectPoints;
    cv::Size imageSize, boardSize;
    float squareSize;
    
    Mat camera1, camera2, dist1, dist2;
    Mat R, T;
    
    //undistortion
    Mat distIR, distRGB, undistIR, undistRGB;
    
    //depth remaped
    Mat depth;
    Mat_<double> P3D;
    Mat_<double> P3Dp;
}

- (BOOL)connectAndStartStreaming;
- (void)renderDepthFrame:(STDepthFrame*)depthFrame;
- (void)renderNormalsFrame:(STDepthFrame*)normalsFrame;
- (void)renderColorFrame:(CMSampleBufferRef)sampleBuffer;
- (void)setupColorCamera;
- (void)startColorCamera;
- (void)stopColorCamera;

@property (weak) NSTimer *repeatingTimer;
- (NSDictionary *)userInfo;

@end

@implementation ViewController

- (void)viewDidLoad
{
    [super viewDidLoad];
    
    _sensorController = [STSensorController sharedController];
    _sensorController.delegate = self;

    // Create three image views where we will render our frames
    
    CGRect depthFrame = self.view.frame;
    depthFrame.size.height /= 2;
    depthFrame.origin.y = self.view.frame.size.height/2;
    depthFrame.origin.x = 1;
    depthFrame.origin.x = -self.view.frame.size.width * 0.25;
    
    CGRect normalsFrame = self.view.frame;
    normalsFrame.size.height /= 2;
    normalsFrame.origin.y = self.view.frame.size.height/2;
    normalsFrame.origin.x = 1;
    normalsFrame.origin.x = self.view.frame.size.width * 0.25;
    
    CGRect colorFrame = self.view.frame;
    colorFrame.size.height /= 2;
    colorFrame.origin.x = 1;
    colorFrame.origin.x = self.view.frame.size.width * 0.25;

    CGRect irFrame = self.view.frame;
    irFrame.size.height /= 2;
    irFrame.origin.x = 1;
    irFrame.origin.x = -self.view.frame.size.width * 0.25;
    
    _linearizeBuffer = NULL;
    _coloredDepthBuffer = NULL;
    _normalsBuffer = NULL;

    _depthImageView = [[UIImageView alloc] initWithFrame:depthFrame];
    _depthImageView.contentMode = UIViewContentModeScaleAspectFit;
    [self.view addSubview:_depthImageView];
    
    _normalsImageView = [[UIImageView alloc] initWithFrame:normalsFrame];
    _normalsImageView.contentMode = UIViewContentModeScaleAspectFit;
    [self.view addSubview:_normalsImageView];
    
    _colorImageView = [[UIImageView alloc] initWithFrame:colorFrame];
    _colorImageView.contentMode = UIViewContentModeScaleAspectFit;
    [self.view addSubview:_colorImageView];

    _irImageView = [[UIImageView alloc] initWithFrame:irFrame];
    _irImageView.contentMode = UIViewContentModeScaleAspectFit;
    [self.view addSubview:_irImageView];

    [self setupColorCamera];
    
    //take listener
    UITapGestureRecognizer *singleFingerTap =
    [[UITapGestureRecognizer alloc] initWithTarget:self
                                            action:@selector(handleSingleTap:)];
    [self.view addGestureRecognizer:singleFingerTap];
    
    counter = 0;
    sequence = false;
    recording = false;
    
    // calibration stuff
    streaming = true;
    makeNew = true;
    useUndist = false;
    remap3D = false;
    
    boardSize.width = 9;
    boardSize.height = 6;
    squareSize = 2.5f;
    P3D = cv::Mat(3,1, CV_32F);
    
    //calibration labels and buttons
    // custom debug labels
    _calibCounter = [[UILabel alloc] initWithFrame:self.view.bounds];
    _calibCounter.frame = CGRectMake(self.view.frame.size.width-150, self.view.frame.size.height-40, 150.0, 40.0);
    _calibCounter.font = [UIFont systemFontOfSize:15.0];
    _calibCounter.textColor = [UIColor redColor];
    _calibCounter.text = @"calibrate images";
    [self.view addSubview: _calibCounter];
    
    
    _text2 = [[UILabel alloc] initWithFrame:self.view.bounds];
    _text2.frame = CGRectMake(self.view.frame.size.width-200, self.view.frame.size.height-80, 200.0, 40.0);
    _text2.font = [UIFont systemFontOfSize:15.0];
    _text2.numberOfLines = 2;
    _text2.textColor = [UIColor redColor];
    [self.view addSubview: _text2];
    
    _text3 = [[UILabel alloc] initWithFrame:self.view.bounds];
    _text3.frame = CGRectMake(self.view.frame.size.width-200, self.view.frame.size.height-120, 200.0, 40.0);
    _text3.font = [UIFont systemFontOfSize:15.0];
    _text3.numberOfLines = 2;
    _text3.textColor = [UIColor redColor];
    [self.view addSubview: _text3];
    
    _text4 = [[UILabel alloc] initWithFrame:self.view.bounds];
    _text4.frame = CGRectMake(self.view.frame.size.width-200, self.view.frame.size.height-160, 200.0, 40.0);
    _text4.font = [UIFont systemFontOfSize:15.0];
    _text4.numberOfLines = 2;
    _text4.textColor = [UIColor redColor];
    [self.view addSubview: _text4];
    
    _text5 = [[UILabel alloc] initWithFrame:self.view.bounds];
    _text5.frame = CGRectMake(self.view.frame.size.width-200, self.view.frame.size.height-200, 200.0, 40.0);
    _text5.font = [UIFont systemFontOfSize:15.0];
    _text5.numberOfLines = 2;
    _text5.textColor = [UIColor redColor];
    [self.view addSubview: _text5];
    
    //right side buttons
    
    _calib = [UIButton buttonWithType:UIButtonTypeRoundedRect];
    [_calib addTarget:self
               action:@selector(calibrate:)
     forControlEvents:UIControlEventTouchUpInside];
    [_calib setTitle:@"Calibrate cameras" forState:UIControlStateNormal];
    _calib.frame = CGRectMake(self.view.frame.size.width-160, 20, 160.0, 50.0);
    [self.view addSubview:_calib];
    
    _stereoCalib = [UIButton buttonWithType:UIButtonTypeRoundedRect];
    [_stereoCalib addTarget:self
               action:@selector(stereoCalibrate:)
     forControlEvents:UIControlEventTouchUpInside];
    [_stereoCalib setTitle:@"Stereo calibrate" forState:UIControlStateNormal];
    _stereoCalib.frame = CGRectMake(self.view.frame.size.width-160, 80, 160.0, 50.0);
    [self.view addSubview:_stereoCalib];
    
    _restartStream = [UIButton buttonWithType:UIButtonTypeRoundedRect];
    [_restartStream addTarget:self
                     action:@selector(restartStream:)
           forControlEvents:UIControlEventTouchUpInside];
    [_restartStream setTitle:@"Continue calibration" forState:UIControlStateNormal];
    _restartStream.frame = CGRectMake(self.view.frame.size.width-160, 140, 160.0, 50.0);
    [self.view addSubview:_restartStream];
    
    _toggleDist = [UIButton buttonWithType:UIButtonTypeRoundedRect];
    [_toggleDist addTarget:self
                       action:@selector(toggleDist:)
             forControlEvents:UIControlEventTouchUpInside];
    [_toggleDist setTitle:@"Toggle undistortion" forState:UIControlStateNormal];
    _toggleDist.frame = CGRectMake(self.view.frame.size.width-160, 200, 160.0, 50.0);
    [self.view addSubview:_toggleDist];
    
    _remapDepth = [UIButton buttonWithType:UIButtonTypeRoundedRect];
    [_remapDepth addTarget:self
                       action:@selector(remapDepth:)
             forControlEvents:UIControlEventTouchUpInside];
    [_remapDepth setTitle:@"Show remapped depth" forState:UIControlStateNormal];
    _remapDepth.frame = CGRectMake(self.view.frame.size.width-160, 260, 160.0, 50.0);
    [self.view addSubview:_remapDepth];
    
    _recordSeq = [UIButton buttonWithType:UIButtonTypeRoundedRect];
    [_recordSeq addTarget:self
                    action:@selector(recordSeq:)
          forControlEvents:UIControlEventTouchUpInside];
    [_recordSeq setTitle:@"toggle recording" forState:UIControlStateNormal];
    _recordSeq.frame = CGRectMake(self.view.frame.size.width-160, 320, 200.0, 50.0);
    [self.view addSubview:_recordSeq];
    
    _saveCalib = [UIButton buttonWithType:UIButtonTypeRoundedRect];
    [_saveCalib addTarget:self
                   action:@selector(saveCalibData:)
         forControlEvents:UIControlEventTouchUpInside];
    [_saveCalib setTitle:@"save calibration" forState:UIControlStateNormal];
    _saveCalib.frame = CGRectMake(self.view.frame.size.width-160, 380, 200.0, 50.0);
    [self.view addSubview:_saveCalib];
    
    _recordSeqAuto = [UIButton buttonWithType:UIButtonTypeRoundedRect];
    [_recordSeqAuto addTarget:self
                   action:@selector(saveSequence:)
         forControlEvents:UIControlEventTouchUpInside];
    [_recordSeqAuto setTitle:@"start/stop recording sequence" forState:UIControlStateNormal];
    _recordSeqAuto.frame = CGRectMake(self.view.frame.size.width-160, 440, 200.0, 50.0);
    [self.view addSubview:_recordSeqAuto];
    
    
    //left side buttons
    
    _showIR = [UIButton buttonWithType:UIButtonTypeRoundedRect];
    [_showIR addTarget:self
                       action:@selector(showIR:)
             forControlEvents:UIControlEventTouchUpInside];
    [_showIR setTitle:@"Show IR" forState:UIControlStateNormal];
    _showIR.frame = CGRectMake(0, 20, 160.0, 50.0);
    [self.view addSubview:_showIR];
    
    _showDepth = [UIButton buttonWithType:UIButtonTypeRoundedRect];
    [_showDepth addTarget:self
                       action:@selector(showDepth:)
             forControlEvents:UIControlEventTouchUpInside];
    [_showDepth setTitle:@"Show depth" forState:UIControlStateNormal];
    _showDepth.frame = CGRectMake(0, 80, 160.0, 50.0);
    [self.view addSubview:_showDepth];
    
    _showBoth = [UIButton buttonWithType:UIButtonTypeRoundedRect];
    [_showBoth addTarget:self
                   action:@selector(showBoth:)
         forControlEvents:UIControlEventTouchUpInside];
    [_showBoth setTitle:@"Show both (slow)" forState:UIControlStateNormal];
    _showBoth.frame = CGRectMake(0, 140, 160.0, 50.0);
    [self.view addSubview:_showBoth];
    
    _showIRSync = [UIButton buttonWithType:UIButtonTypeRoundedRect];
    [_showIRSync addTarget:self
                action:@selector(showIR:)
      forControlEvents:UIControlEventTouchUpInside];
    [_showIRSync setTitle:@"Show IR sync" forState:UIControlStateNormal];
    _showIRSync.frame = CGRectMake(0, 200, 160.0, 50.0);
    [self.view addSubview:_showIRSync];
    
    _showDepthSync = [UIButton buttonWithType:UIButtonTypeRoundedRect];
    [_showDepthSync addTarget:self
                   action:@selector(showDepthSync:)
         forControlEvents:UIControlEventTouchUpInside];
    [_showDepthSync setTitle:@"Show depth sync" forState:UIControlStateNormal];
    _showDepthSync.frame = CGRectMake(0, 260, 160.0, 50.0);
    [self.view addSubview:_showDepthSync];
    
}

- (void)dealloc
{
    if (_linearizeBuffer)
        free(_linearizeBuffer);
    
    if (_coloredDepthBuffer)
        free(_coloredDepthBuffer);
    
    if (_normalsBuffer)
        free(_normalsBuffer);
}

- (void)viewDidAppear:(BOOL)animated
{
    static BOOL fromLaunch = true;
    if(fromLaunch)
    {

        //
        // Create a UILabel in the center of our view to display status messages
        //
    
        // We do this here instead of in viewDidLoad so that we get the correctly size/rotation view bounds
        if (!_statusLabel) {
            
            _statusLabel = [[UILabel alloc] initWithFrame:self.view.bounds];
            _statusLabel.backgroundColor = [[UIColor blackColor] colorWithAlphaComponent:0.7];
            _statusLabel.textAlignment = NSTextAlignmentCenter;
            _statusLabel.font = [UIFont systemFontOfSize:35.0];
            _statusLabel.numberOfLines = 2;
            _statusLabel.textColor = [UIColor whiteColor];

            [self updateAppStatusMessage];
            
            [self.view addSubview: _statusLabel];
            
        }

        [self connectAndStartStreaming];
        fromLaunch = false;

        // From now on, make sure we get notified when the app becomes active to restore the sensor state if necessary.
        [[NSNotificationCenter defaultCenter] addObserver:self
                                                 selector:@selector(appDidBecomeActive)
                                                     name:UIApplicationDidBecomeActiveNotification
                                                   object:nil];
    }
}


- (void)appDidBecomeActive
{
    [self connectAndStartStreaming];
}


- (void)didReceiveMemoryWarning
{
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

- (NSDictionary *)userInfo {
    return @{ @"StartDate" : [NSDate date] };
}

- (BOOL)connectAndStartStreaming
{
    
    STSensorControllerInitStatus result = [_sensorController initializeSensorConnection];
    
    BOOL didSucceed = (result == STSensorControllerInitStatusSuccess || result == STSensorControllerInitStatusAlreadyInitialized);
    
    
    if (didSucceed)
    {
        // There's no status about the sensor that we need to display anymore
        _appStatus.sensorStatus = AppStatus::SensorStatusOk;
        [self updateAppStatusMessage];
        
        // Start the color camera, setup if needed
        [self startColorCamera];
        
        // Set sensor stream quality
        STStreamConfig streamConfig = STStreamConfigInfrared640x488;//STStreamConfigDepth640x480;STStreamConfigDepth320x240AndInfrared320x248;
        
        // Request that we receive depth frames with synchronized color pairs
        // After this call, we will start to receive frames through the delegate methods
        NSError* error = nil;
        BOOL optionsAreValid = [_sensorController startStreamingWithOptions:@{kSTStreamConfigKey : @(streamConfig),
                                                                              } error:&error];
        
        
        if (!optionsAreValid)
        {
            [self showAppStatusMessage:@"invalid options"];
            NSLog(@"Error during streaming start: %s", [[error localizedDescription] UTF8String]);
            return false;
        }
        
        // Allocate the depth (shift) -> to depth (millimeters) converter class
        _floatDepthFrame = [[STFloatDepthFrame alloc] init];
        _lastFloatDepth = [[STFloatDepthFrame alloc] init];

        // allocate last received frames -> to save them somewhere
        _lastColorImage = [[UIImage alloc] init];
        _lastInfraredImage = [[UIImage alloc] init];
//        _lastDepthFrame = [[STFrame alloc] init];


        // Allocate the depth -> surface normals converter class
        _normalsEstimator = [[STNormalEstimator alloc] initWithSensorInfo:[_sensorController getSensorInfo:streamConfig]];
    }
    else
    {
        if (result == STSensorControllerInitStatusSensorNotFound)
            NSLog(@"[Debug] No Structure Sensor found!");
        else if (result == STSensorControllerInitStatusOpenFailed)
            NSLog(@"[Error] Structure Sensor open failed.");
        else if (result == STSensorControllerInitStatusSensorIsWakingUp)
            NSLog(@"[Debug] Structure Sensor is waking from low power.");
        else if (result != STSensorControllerInitStatusSuccess)
            NSLog(@"[Debug] Structure Sensor failed to init with status %d.", (int)result);
        
        _appStatus.sensorStatus = AppStatus::SensorStatusNeedsUserToConnect;
        [self updateAppStatusMessage];
    }
    
    return didSucceed;
    
}

- (void)showAppStatusMessage:(NSString *)msg
{
    _appStatus.needsDisplayOfStatusMessage = true;
    [self.view.layer removeAllAnimations];
    
    [_statusLabel setText:msg];
    [_statusLabel setHidden:NO];
    
    // Progressively show the message label.
    [self.view setUserInteractionEnabled:false];
    [UIView animateWithDuration:0.5f animations:^{
        _statusLabel.alpha = 1.0f;
    }completion:nil];
}

- (void)hideAppStatusMessage
{
    
    _appStatus.needsDisplayOfStatusMessage = false;
    [self.view.layer removeAllAnimations];
    
    [UIView animateWithDuration:1.0f
                     animations:^{
                         _statusLabel.alpha = 0.0f;
                     }
                     completion:^(BOOL finished) {
                         // If nobody called showAppStatusMessage before the end of the animation, do not hide it.
                         if (!_appStatus.needsDisplayOfStatusMessage)
                         {
                             [_statusLabel setHidden:YES];
                             [self.view setUserInteractionEnabled:true];
                         }
                     }];
}

-(void)updateAppStatusMessage
{
    // Skip everything if we should not show app status messages (e.g. in viewing state).
    if (_appStatus.statusMessageDisabled)
    {
        [self hideAppStatusMessage];
        return;
    }
    
    // First show sensor issues, if any.
    switch (_appStatus.sensorStatus)
    {
        case AppStatus::SensorStatusOk:
        {
            break;
        }
            
        case AppStatus::SensorStatusNeedsUserToConnect:
        {
            [self showAppStatusMessage:_appStatus.pleaseConnectSensorMessage];
            return;
        }
            
        case AppStatus::SensorStatusNeedsUserToCharge:
        {
            [self showAppStatusMessage:_appStatus.pleaseChargeSensorMessage];
            return;
        }
    }
    
    // Then show color camera permission issues, if any.
    if (!_appStatus.colorCameraIsAuthorized)
    {
        [self showAppStatusMessage:_appStatus.needColorCameraAccessMessage];
        return;
    }
    
    // If we reach this point, no status to show.
    [self hideAppStatusMessage];
}

-(bool) isConnectedAndCharged
{
    return [_sensorController isConnected] && ![_sensorController isLowPower];
}

#pragma mark -
#pragma mark Structure SDK Delegate Methods

- (void)sensorDidDisconnect
{
    NSLog(@"Structure Sensor disconnected!");

    _appStatus.sensorStatus = AppStatus::SensorStatusNeedsUserToConnect;
    [self updateAppStatusMessage];
    
    // Stop the color camera when there isn't a connected Structure Sensor
    [self stopColorCamera];
}

- (void)sensorDidConnect
{
    NSLog(@"Structure Sensor connected!");
    [self connectAndStartStreaming];
}

- (void)sensorDidLeaveLowPowerMode
{
    _appStatus.sensorStatus = AppStatus::SensorStatusNeedsUserToConnect;
    [self updateAppStatusMessage];
}


- (void)sensorBatteryNeedsCharging
{
    // Notify the user that the sensor needs to be charged.
    _appStatus.sensorStatus = AppStatus::SensorStatusNeedsUserToCharge;
    [self updateAppStatusMessage];
}

- (void)sensorDidStopStreaming:(STSensorControllerDidStopStreamingReason)reason
{
    //If needed, change any UI elements to account for the stopped stream

    // Stop the color camera when we're not streaming from the Structure Sensor
    [self stopColorCamera];

}

- (void)sensorDidOutputDepthFrame:(STDepthFrame *)depthFrame
{
    if(streaming)
    {
        [self renderDepthFrame: depthFrame];
        [self renderNormalsFrame: depthFrame];
        [_lastFloatDepth updateFromDepthFrame: depthFrame];
        
        if(useUndist) {
            _lastColorImage = [self undistRGB:_lastColorImage];
        }
        _colorImageView.image = _lastColorImage;
    }
}

- (void)sensorDidOutputInfraredFrame:(STInfraredFrame *)irFrame
{
    if(streaming)
    {
        [self renderIRFrame:irFrame];
        _lastInfraredImage = [self UIImageFromIRFrame: irFrame];
        
        if(useUndist) {
            _lastColorImage = [self undistRGB:_lastColorImage];
        }
        _colorImageView.image = _lastColorImage;
    }
}

- (void)sensorDidOutputSynchronizedDepthFrame:(STDepthFrame*)depthFrame
                               andColorBuffer:(CMSampleBufferRef)sampleBuffer
{
    if(streaming)
    {
        [self renderColorFrame: sampleBuffer];
        [self renderDepthFrame: depthFrame];
        [self renderNormalsFrame: depthFrame];
        [_lastFloatDepth updateFromDepthFrame: depthFrame];
    }
}

- (void)sensorDidOutputSynchronizedIRFrame:(STInfraredFrame *)    irFrame
                            andColorFrame:(CMSampleBufferRef)sampleBuffer;
{
    if(streaming)
    {
        [self renderColorFrame: sampleBuffer];
        [self renderIRFrame:irFrame];
        _lastInfraredImage = [self UIImageFromIRFrame: irFrame];
    }
}

- (void)captureOutput:(AVCaptureOutput *)captureOutput didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection *)connection
{
    // Pass into the driver. The sampleBuffer will return later with a synchronized depth or IR pair.
    
    if(streaming)
    {
        //[self renderColorFrame: sampleBuffer];
        [_sensorController frameSyncNewColorBuffer:sampleBuffer];
        _lastColorImage = [self UIImageFromBuffer:sampleBuffer];
    }
}

- (void) restartStream: (id) sender
{
    streaming = true;
}

- (void) showIR: (id) sender
{
    // Set sensor stream quality
    STStreamConfig streamConfig = STStreamConfigInfrared640x488;
    
    // Request that we receive depth frames with synchronized color pairs
    // After this call, we will start to receive frames through the delegate methods
    NSError* error = nil;
    [_sensorController startStreamingWithOptions:@{kSTStreamConfigKey : @(streamConfig),
                                                                          } error:&error];

}

- (void) showDepth: (id) sender
{
    // Set sensor stream quality
    STStreamConfig streamConfig = STStreamConfigDepth640x480;//STStreamConfigDepth320x240AndInfrared320x248;
    
    // Request that we receive depth frames with synchronized color pairs
    // After this call, we will start to receive frames through the delegate methods
    NSError* error = nil;
    [_sensorController startStreamingWithOptions:@{kSTStreamConfigKey : @(streamConfig),
                                                                          } error:&error];

}

- (void) showIRSync: (id) sender
{
    
    // Set sensor stream quality
    STStreamConfig streamConfig = STStreamConfigInfrared320x248;
    
    // Request that we receive depth frames with synchronized color pairs
    // After this call, we will start to receive frames through the delegate methods
    NSError* error = nil;
    [_sensorController startStreamingWithOptions:@{kSTStreamConfigKey : @(streamConfig),
                                                   kSTFrameSyncConfigKey : @(STFrameSyncInfraredAndRgb)
                                                   } error:&error];
    
}

- (void) showDepthSync: (id) sender
{
    // Set sensor stream quality
    STStreamConfig streamConfig = STStreamConfigDepth640x480;// STStreamConfigDepth320x240;//STStreamConfigDepth320x240AndInfrared320x248;
    
    // Request that we receive depth frames with synchronized color pairs
    // After this call, we will start to receive frames through the delegate methods
    NSError* error = nil;
    [_sensorController startStreamingWithOptions:@{kSTStreamConfigKey : @(streamConfig),
                                                   kSTFrameSyncConfigKey : @(STFrameSyncDepthAndRgb)
                                                   } error:&error];
    
}

- (void) showBoth: (id) sender
{
    // Set sensor stream quality
    STStreamConfig streamConfig = STStreamConfigDepth640x480AndInfrared640x488;// STStreamConfigDepth320x240AndInfrared320x248;
    
    // Request that we receive depth frames with synchronized color pairs
    // After this call, we will start to receive frames through the delegate methods
    NSError* error = nil;
    [_sensorController startStreamingWithOptions:@{kSTStreamConfigKey : @(streamConfig),
                                                   } error:&error];
    
}

- (void)handleSingleTap:(UITapGestureRecognizer *)recognizer {
    //CGPoint location = [recognizer locationInView:[recognizer.view superview]];
    
//    UIImage* img = _lastColorImage;
//    UIImage* ir = _lastInfraredImage;
//    
//    UIImageWriteToSavedPhotosAlbum(img, nil, nil, nil);
//    UIImageWriteToSavedPhotosAlbum(ir, nil, nil, nil);
    
    streaming = false;
    
    if(sequence) {
        [self saveDepthAndRGB];
    } else {
        [self findCorners];
    }
    
    streaming = true;
}

-(void) saveDepthAndRGB
{
    
    UIImage* img = _lastColorImage;
//    Mat image = [self cvMatFromUIImage: img];
    
    void * ptr = (void*)_lastFloatDepth.depthAsMillimeters;
    Mat cvMat = Mat(_lastFloatDepth.height, _lastFloatDepth.width, CV_32FC1, ptr);
    
    NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *docs = [paths objectAtIndex:0];
    
    NSString *name = [NSString stringWithFormat:@"sequence%d.xml",counter];
    NSString *vocabPath = [docs stringByAppendingPathComponent:name];
    FileStorage fs([vocabPath UTF8String], FileStorage::WRITE);
    
//    fs << "depth" << cvMat << "image" << image;
    fs << "depth" << cvMat;
    
    //build the path for your image on the filesystem
    NSString *photoName = [NSString stringWithFormat:@"sequence%d.png",counter];
    NSString *photoPath = [docs stringByAppendingPathComponent:photoName];
    NSData *imageData = UIImagePNGRepresentation(img);
    
    [imageData writeToFile:photoPath atomically:YES];
    
    [_text2 setText:[NSString stringWithFormat:@"%d images saved",counter]];
    
    fs.release();
    counter = counter+1;
    
}

-(void) saveDepthAndRGBcaller: (id) sender
{
    [self saveDepthAndRGB];
}

-(void) saveSequence: (id) sender
{
    
    if (recording) {
        //cancel timer
        [_text4 setText:@"not recording"];
        [self.repeatingTimer invalidate];
        recording = !recording;
    } else {
        [_text4 setText:@"recording"];
        NSTimer *timer = [NSTimer scheduledTimerWithTimeInterval:0.1
                                                          target:self
                                                          selector: @selector(saveDepthAndRGBcaller:)
                                                          userInfo:nil
                                                          repeats:YES];
        self.repeatingTimer = timer;
        recording = !recording;
    }
}

-(void) saveCalibData: (id) sender
{
    NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *docs = [paths objectAtIndex:0];
    
    NSString *vocabPath = [docs stringByAppendingPathComponent:@"calib.xml"];
    FileStorage fs([vocabPath UTF8String], FileStorage::WRITE);
    fs << "camera1" << camera1 << "camera2" << camera2 << "T" << T << "R" << R;
    fs.release();
}

- (void) findCorners
{
    
    UIImage* UIrgb = _lastColorImage;
    UIImage* UIir = _lastInfraredImage;
    
    Mat rgb = [self cvMatFromUIImage:UIrgb];
    cvtColor(rgb,rgb,CV_BGR2GRAY);
    Mat ir = [self cvMatGrayFromUIImage:UIir];
    
    imageSize = rgb.size();
    resize(ir,ir,imageSize);
    
    _normalsImageView.image = [self UIImageFromCVMat: rgb];
    _depthImageView.image = [self UIImageFromCVMat: ir];
    
    int j = imagePoints[0].size();
    if(makeNew) {
        imagePoints[0].resize(j+1);
        imagePoints[1].resize(j+1);
        j++;
        makeNew = false;
    }
    
    bool found = false;
    
    vector<Point2f>& cornersIR = imagePoints[0][j-1];
    found = findChessboardCorners(ir, boardSize, cornersIR,
                                    CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);
    
    // corners in ir image haven't been found
    if( found ) {
        //[self showAppStatusMessage:@"IR chessboard found"];
    } else {
        [self showAppStatusMessage:@"IR chessboard not found"];
        [self hideAppStatusMessage];
        return;
    }
    
    cornerSubPix(ir, cornersIR, cv::Size(11,11), cv::Size(-1,-1),
                 TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,
                              30, 0.01));
    
    vector<Point2f>& cornersRGB = imagePoints[1][j-1];
    found = findChessboardCorners(rgb, boardSize, cornersRGB,
                                  CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);
    
    //corners in rgb image found
    if( found ) {
        //[self showAppStatusMessage:@"RGB chessboard found"];
    } else {
        [self showAppStatusMessage:@"RGB chessboard not found"];
        [self hideAppStatusMessage];
        return;
    }
    
    cornerSubPix(rgb, cornersRGB, cv::Size(11,11), cv::Size(-1,-1),
                 TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,
                              30, 0.01));
    
    // corner detection has been successful in both images
    [self showAppStatusMessage:@"both chessboards found"];
    [self hideAppStatusMessage];
    
    // after all display the corners
    Mat timg1, timg2;
    cvtColor(ir, timg1, COLOR_GRAY2BGR);
    cvtColor(rgb, timg2, COLOR_GRAY2BGR);
    drawChessboardCorners(timg1, boardSize, cornersIR, found);
    drawChessboardCorners(timg2, boardSize, cornersRGB, found);
    
    _normalsImageView.image = [self UIImageFromCVMat: timg2];
    _depthImageView.image = [self UIImageFromCVMat: timg1];
    
    _calibCounter.text = [NSString stringWithFormat: @"Recorded images: %lu", imagePoints[0].size()];
    
    makeNew = true;
    
}

- (void) calibrate: (id) sender
{
    int nimages = imagePoints[0].size();
    
    if(nimages<2) {
        [self showAppStatusMessage: [NSString stringWithFormat: @"not enough image pairs: %d", nimages]];
        [self hideAppStatusMessage];
        return;
    }
    
    // if makeNew is false, the last entry of imagePoints should be deleted
    if(!makeNew) {
        nimages--;
        imagePoints[0].resize(nimages);
        imagePoints[1].resize(nimages);
        makeNew = true;
    }
    
    objectPoints.clear();
    objectPoints.resize(nimages);
    
    //create the object points for the calibration
    for( int i = 0; i < nimages; i++ )
    {
        for( int j = 0; j < boardSize.height; j++ )
            for( int k = 0; k < boardSize.width; k++ )
                objectPoints[i].push_back(Point3f(
                                                  float(k*squareSize),
                                                  float(j*squareSize),
                                                  0));
    }

    vector<Mat> rvecs, tvecs;
    double rms;

    //IR calib
    rms = calibrateCamera(objectPoints, imagePoints[0], imageSize, camera1,
                                 dist1, rvecs, tvecs, CV_CALIB_FIX_PRINCIPAL_POINT);
    [_text2 setText: [NSString stringWithFormat: @"RMS error IR: %f", rms]];
    
    //RGB calib
    rms = calibrateCamera(objectPoints, imagePoints[1], imageSize, camera2,
                          dist2, rvecs, tvecs, CV_CALIB_FIX_PRINCIPAL_POINT);
    [_text3 setText: [NSString stringWithFormat: @"RMS error RGB: %f", rms]];
    
    useUndist = true;
    
}

- (void) stereoCalibrate: (id) sender
{
    
    if(!useUndist) {
        [self showAppStatusMessage: @"do single camera calibration first"];
        [self hideAppStatusMessage];
        return;
    }
    [self showAppStatusMessage: @"do stereo calibration"];
    [self hideAppStatusMessage];
    
    Mat E, F;
    double rms = stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],
                                 camera1, dist1,
                                 camera2, dist2,
                                 imageSize, R, T, E, F,
                                 TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 50, 1e-6),
                                 CV_CALIB_FIX_INTRINSIC +
                                 CV_CALIB_USE_INTRINSIC_GUESS +
                                 CV_CALIB_FIX_ASPECT_RATIO +
                                 CV_CALIB_ZERO_TANGENT_DIST +
                                 CV_CALIB_SAME_FOCAL_LENGTH +
                                 CV_CALIB_RATIONAL_MODEL
                                 + CV_CALIB_FIX_K3 + CV_CALIB_FIX_K4 + CV_CALIB_FIX_K5
                                 );
    
    [_text4 setText: [NSString stringWithFormat: @"RMS error for stereo: %f", rms]];
    
}

- (void) toggleDist: (id) sender
{
    useUndist = !useUndist;
}

- (void) remapDepth: (id) sender
{
    remap3D = !remap3D;
}

- (void) recordSeq: (id) sender
{
    sequence = !sequence;
}

- (void) showRectification
{
    
    int nimages = objectPoints.size();
    Mat R1, R2, P1, P2, Q;
    cv::Rect validRoi[2];
    
    stereoRectify(camera1, dist1,
                  camera2, dist2,
                  imageSize, R, T, R1, R2, P1, P2, Q,
                  CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]);

    
    Mat rmap[2][2];
    vector<Point2f> allimgpt[2];
    for( int k = 0; k < 2; k++ )
    {
        for( int i = 0; i < nimages; i++ )
            std::copy(imagePoints[k][i].begin(), imagePoints[k][i].end(), back_inserter(allimgpt[k]));
    }
    Mat F = findFundamentalMat(Mat(allimgpt[0]), Mat(allimgpt[1]), FM_8POINT, 0, 0);
    Mat H1, H2;
    stereoRectifyUncalibrated(Mat(allimgpt[0]), Mat(allimgpt[1]), F, imageSize, H1, H2, 3);
    
    R1 = camera1.inv()*H1*camera1;
    R2 = camera2.inv()*H2*camera2;
    P1 = camera1;
    P2 = camera2;
    
    initUndistortRectifyMap(camera1, dist1, R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
    initUndistortRectifyMap(camera2, dist2, R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);
    
    //get an image and display it
    
    UIImage* UIrgb = _lastColorImage;
    UIImage* UIir = _lastInfraredImage;
    
    Mat rgb = [self cvMatFromUIImage:UIrgb];
    cvtColor(rgb,rgb,CV_BGR2GRAY);
    Mat ir = [self cvMatGrayFromUIImage:UIir];
    
    imageSize = rgb.size();
    resize(ir,ir,imageSize);

    Mat rectIR, rectRGB, trectIR, trectRGB;
    remap(ir, trectIR, rmap[0][0], rmap[0][1], CV_INTER_LINEAR);
    cvtColor(trectIR, rectIR, COLOR_GRAY2BGR);
    
    remap(rgb, trectRGB, rmap[0][0], rmap[0][1], CV_INTER_LINEAR);
    cvtColor(trectRGB, rectRGB, COLOR_GRAY2BGR);
    
    _normalsImageView.image = [self UIImageFromCVMat: rectRGB];
    _depthImageView.image = [self UIImageFromCVMat: rectIR];
}

- (Mat) convertDepthMap: (const float *) depthMap
{
    Mat a;
    
//    NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
//    NSString *docs = [paths objectAtIndex:0];
//    
//    NSString *vocabPath = [docs stringByAppendingPathComponent:@"prop.xml"];
//    FileStorage fs([vocabPath UTF8String], FileStorage::WRITE);
//    fs << "camera1" << camera1 << "camera1" << camera2;
//    
//    fs.release();
    
    Mat depthmap_rgbcam = cv::Mat::zeros(imageSize.height,imageSize.width,CV_32FC1); // Initialize the depthmap to all zeros
    float *depthmap_rgbcam_buffer = (float*)depthmap_rgbcam.data;
    
    streaming = false;
    
//    for( int row = 0; row < imageSize.height; row++ ) {
//        for( int col = 0; col < imageSize.width; col++ ) {
    for( int row = 0; row < imageSize.height/2; row++ ) {
        for( int col = 0; col < imageSize.width/2; col++ ) {
                    
            
            float depth = depthMap[imageSize.width*row + col];
            float depth_depthcam = depth;
            
//            P_ir = inv(K) * p_ir

//            P3D.x = (x_d - cx_d) * depth(x_d,y_d) / fx_d
//            P3D.y = (y_d - cy_d) * depth(x_d,y_d) / fy_d
//            P3D.z = depth(x_d,y_d)
            
            // Map depthcam depth to 3D point
            P3D(0) = (col - camera1.at<double>(0,2)) * depth / camera1.at<double>(0,0);
            P3D(1) = (row - camera1.at<double>(1,2)) * depth / camera1.at<double>(1,1);
            P3D(2) = depth;
            
//            P_rgb = R * P_ir + t
//            
//            P3D' = R.P3D + T
            
            // Rotate and translate 3D point
            P3Dp = (R*P3D) + T;
            
//            p_rgb = K_rgb * P_rgb
            
//            P2D_rgb.x = (P3D'.x * fx_rgb / P3D'.z) + cx_rgb
//            P2D_rgb.y = (P3D'.y * fy_rgb / P3D'.z) + cy_rgb
            
            // Project 3D point to rgbcam
            float xrgb = (P3Dp(0) * camera2.at<double>(0,0) / P3Dp(2)) + camera2.at<double>(0,2);
            float yrgb = (P3Dp(1) * camera2.at<double>(1,1) / P3Dp(2)) + camera2.at<double>(1,2);
            
            // "Interpolate" pixel coordinates (Nearest Neighbors)
            int px_rgbcam = cvRound(xrgb);
            int py_rgbcam = cvRound(yrgb);
            
            // Handle 3D occlusions
            
            
            float &depth_rgbcam = depthmap_rgbcam_buffer[imageSize.width*py_rgbcam+px_rgbcam];

             if(depth_rgbcam==0 || depth_depthcam<depth_rgbcam)
                 depth_rgbcam = depth_depthcam;
            
        }
        [_text5 setText:[NSString stringWithFormat:@"row: %d", row]];
    }
    
    streaming = true;
    
    return depthmap_rgbcam;
}

- (UIImage *) undistRGB: (UIImage*) image
{
    distRGB = [self cvMatFromUIImage:image];
    undistort(distRGB, undistRGB, camera2, dist2);
    image = [self UIImageFromCVMat:undistRGB];
    return image;
}

- (UIImage *) undistIR: (UIImage*) image
{
    distIR = [self cvMatGrayFromUIImage:image];
    undistort(distIR, undistIR, camera1, dist1);
    image = [self UIImageFromCVMat:undistIR];
    return image;
}

- (cv::Mat)cvMatGrayFromUIImage:(UIImage *)image
{
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
    CGFloat cols = image.size.width;
    CGFloat rows = image.size.height;
    
    cv::Mat cvMat(rows, cols, CV_8UC1); // 8 bits per component, 1 channels
    
    CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,                 // Pointer to data
                                                    cols,                       // Width of bitmap
                                                    rows,                       // Height of bitmap
                                                    8,                          // Bits per component
                                                    cvMat.step[0],              // Bytes per row
                                                    colorSpace,                 // Colorspace
                                                    kCGImageAlphaNone |
                                                    kCGBitmapByteOrderDefault); // Bitmap info flags
    
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
    CGContextRelease(contextRef);
    
    return cvMat;
}

-(UIImage *)UIImageFromCVMat:(cv::Mat)cvMat
{
    NSData *data = [NSData dataWithBytes:cvMat.data length:cvMat.elemSize()*cvMat.total()];
    CGColorSpaceRef colorSpace;
    
    if (cvMat.elemSize() == 1) {
        colorSpace = CGColorSpaceCreateDeviceGray();
    } else {
        colorSpace = CGColorSpaceCreateDeviceRGB();
    }
    
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    
    // Creating CGImage from cv::Mat
    CGImageRef imageRef = CGImageCreate(cvMat.cols,                                 //width
                                        cvMat.rows,                                 //height
                                        8,                                          //bits per component
                                        8 * cvMat.elemSize(),                       //bits per pixel
                                        cvMat.step[0],                            //bytesPerRow
                                        colorSpace,                                 //colorspace
                                        kCGImageAlphaNone|kCGBitmapByteOrderDefault,// bitmap info
                                        provider,                                   //CGDataProviderRef
                                        NULL,                                       //decode
                                        false,                                      //should interpolate
                                        kCGRenderingIntentDefault                   //intent
                                        );
    
    
    // Getting UIImage from CGImage
    UIImage *finalImage = [UIImage imageWithCGImage:imageRef];
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    
    return finalImage;
}

- (cv::Mat)cvMatFromUIImage:(UIImage *)image
{
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
    CGFloat cols = image.size.width;
    CGFloat rows = image.size.height;
    
    cv::Mat cvMat(rows, cols, CV_8UC4); // 8 bits per component, 4 channels (color channels + alpha)
    
    CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,                 // Pointer to  data
                                                    cols,                       // Width of bitmap
                                                    rows,                       // Height of bitmap
                                                    8,                          // Bits per component
                                                    cvMat.step[0],              // Bytes per row
                                                    colorSpace,                 // Colorspace
                                                    kCGImageAlphaNoneSkipLast |
                                                    kCGBitmapByteOrderDefault); // Bitmap info flags
    
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
    CGContextRelease(contextRef);
    
    return cvMat;
}

#pragma mark -
#pragma mark Rendering

const uint16_t maxShiftValue = 2048;

- (void)populateLinearizeBuffer
{
    _linearizeBuffer = (uint16_t*)malloc((maxShiftValue + 1) * sizeof(uint16_t));
    
    for (int i=0; i <= maxShiftValue; i++)
    {
        float v = i/ (float)maxShiftValue;
        v = powf(v, 3)* 6;
        _linearizeBuffer[i] = v*6*256;
    }
}

// This function is equivalent to calling [STDepthAsRgba convertDepthFrameToRgba] with the
// STDepthToRgbaStrategyRedToBlueGradient strategy. Not using the SDK here for didactic purposes.
- (void)convertShiftToRGBA:(const uint16_t*)shiftValues depthValuesCount:(size_t)depthValuesCount
{
    for (size_t i = 0; i < depthValuesCount; i++)
    {
        // We should not get higher values than maxShiftValue, but let's stay on the safe side.
        uint16_t boundedShift = std::min (shiftValues[i], maxShiftValue);
        
        // Use a lookup table to make the non-linear input values vary more linearly with metric depth
        int linearizedDepth = _linearizeBuffer[boundedShift];
        
        // Use the upper byte of the linearized shift value to choose a base color
        // Base colors range from: (closest) White, Red, Orange, Yellow, Green, Cyan, Blue, Black (farthest)
        int lowerByte = (linearizedDepth & 0xff);
        
        // Use the lower byte to scale between the base colors
        int upperByte = (linearizedDepth >> 8);
        
        switch (upperByte)
        {
            case 0:
                _coloredDepthBuffer[4*i+0] = 255;
                _coloredDepthBuffer[4*i+1] = 255-lowerByte;
                _coloredDepthBuffer[4*i+2] = 255-lowerByte;
                _coloredDepthBuffer[4*i+3] = 255;
                break;
            case 1:
                _coloredDepthBuffer[4*i+0] = 255;
                _coloredDepthBuffer[4*i+1] = lowerByte;
                _coloredDepthBuffer[4*i+2] = 0;
                break;
            case 2:
                _coloredDepthBuffer[4*i+0] = 255-lowerByte;
                _coloredDepthBuffer[4*i+1] = 255;
                _coloredDepthBuffer[4*i+2] = 0;
                break;
            case 3:
                _coloredDepthBuffer[4*i+0] = 0;
                _coloredDepthBuffer[4*i+1] = 255;
                _coloredDepthBuffer[4*i+2] = lowerByte;
                break;
            case 4:
                _coloredDepthBuffer[4*i+0] = 0;
                _coloredDepthBuffer[4*i+1] = 255-lowerByte;
                _coloredDepthBuffer[4*i+2] = 255;
                break;
            case 5:
                _coloredDepthBuffer[4*i+0] = 0;
                _coloredDepthBuffer[4*i+1] = 0;
                _coloredDepthBuffer[4*i+2] = 255-lowerByte;
                break;
            default:
                _coloredDepthBuffer[4*i+0] = 0;
                _coloredDepthBuffer[4*i+1] = 0;
                _coloredDepthBuffer[4*i+2] = 0;
                break;
        }
    }
}


- (UIImage *) UIImageFromBuffer:(CMSampleBufferRef)sampleBuffer
{
    
    CVImageBufferRef pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    
    CVPixelBufferLockBaseAddress(pixelBuffer, 0);
    
    size_t cols = CVPixelBufferGetWidth(pixelBuffer);
    size_t rows = CVPixelBufferGetHeight(pixelBuffer);
    
    
    unsigned char* ptr = (unsigned char*) CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0);
    
    NSData *data = [[NSData alloc] initWithBytes:ptr length:rows*cols*4];
    
    CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
    
    
    
    CGColorSpaceRef colorSpace;
    
    colorSpace = CGColorSpaceCreateDeviceRGB();
    
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((CFDataRef)data);
    
    CGImageRef imageRef = CGImageCreate(cols,                                       //width
                                        rows,                                       //height
                                        8,                                          //bits per component
                                        8 * 4,                                      //bits per pixel
                                        cols*4,                                     //bytesPerRow
                                        colorSpace,                                 //colorspace
                                        kCGImageAlphaNoneSkipFirst|kCGBitmapByteOrder32Little,// bitmap info
                                        provider,                                   //CGDataProviderRef
                                        NULL,                                       //decode
                                        false,                                      //should interpolate
                                        kCGRenderingIntentDefault                   //intent
                                        );
    
    
    // Getting UIImage from CGImage
    UIImage* image = [[UIImage alloc] initWithCGImage:imageRef];
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    
    return image;
    
}

- (UIImage *) UIImageFromIRFrame:(STInfraredFrame *)irFrame
{
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceGray();
    
    CGBitmapInfo bitmapInfo;
    bitmapInfo = (CGBitmapInfo)kCGImageAlphaNone;
    bitmapInfo |= kCGBitmapByteOrder16Big;
    
    //[self showAppStatusMessage: [NSString stringWithFormat: @"data2 %d",irFrame.data]];
    NSData *data = [NSData dataWithBytes:irFrame.data length:irFrame.width * irFrame.height * sizeof(uint16_t)];
    
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    
    CGImageRef cgImage = CGImageCreate(irFrame.width,
                                       irFrame.height,
                                       16,
                                       16,
                                       irFrame.width * sizeof(uint16_t),
                                       colorSpace,
                                       bitmapInfo,
                                       provider,
                                       NULL,
                                       false,
                                       kCGRenderingIntentDefault);
    
    UIImage* image = [UIImage imageWithCGImage:cgImage];
    CFRelease(provider);
    CFRelease(colorSpace);
    CGImageRelease(cgImage);
    
    return image;
}

- (void)renderIRFrame:(STInfraredFrame *)irFrame
{
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceGray();
    
    CGBitmapInfo bitmapInfo;
    bitmapInfo = (CGBitmapInfo)kCGImageAlphaNone;
    bitmapInfo |= kCGBitmapByteOrder16Big;
    
    NSData *data = [NSData dataWithBytes:irFrame.data length:irFrame.width * irFrame.height * sizeof(uint16_t)];
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    
    CGImageRef cgImage = CGImageCreate(irFrame.width,
                                       irFrame.height,
                                       16,
                                       16,
                                       irFrame.width * sizeof(uint16_t),
                                       colorSpace,
                                       bitmapInfo,
                                       provider,
                                       NULL,
                                       false,
                                       kCGRenderingIntentDefault);
    
    UIImage * image = [UIImage imageWithCGImage:cgImage];
    
    if(useUndist) {
        image = [self undistIR:image];
    }
    
    _irImageView.image = image;
    CFRelease(provider);
    CFRelease(colorSpace);
    CGImageRelease(cgImage);
}

- (void)renderDepthFrame:(STDepthFrame *)depthFrame
{
    if(recording) {
        
    } else if(remap3D) {
//        STFloatDepthFrame *floatDepth =[[STFloatDepthFrame alloc] init];
//        [floatDepth updateFromDepthFrame:depthFrame];
//        
//        depth = [self convertDepthMap: floatDepth.depthAsMillimeters];
//        _depthImageView.image = [self UIImageFromCVMat:depth];
    } else {
    
        size_t cols = depthFrame.width;
        size_t rows = depthFrame.height;
        
        if (_linearizeBuffer == NULL || _normalsBuffer == NULL)
        {
            [self populateLinearizeBuffer];
            _coloredDepthBuffer = (uint8_t*)malloc(cols * rows * 4);
        }
        
        // Conversion of 16-bit non-linear shift depth values to 32-bit RGBA
        //
        // Adapted from: https://github.com/OpenKinect/libfreenect/blob/master/examples/glview.c
        //
        [self convertShiftToRGBA:depthFrame.data depthValuesCount:cols * rows];
        
        CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
        
        CGBitmapInfo bitmapInfo;
        bitmapInfo = (CGBitmapInfo)kCGImageAlphaNoneSkipLast;
        bitmapInfo |= kCGBitmapByteOrder32Big;
        
        NSData *data = [NSData dataWithBytes:_coloredDepthBuffer length:cols * rows * 4];
        CGDataProviderRef provider = CGDataProviderCreateWithCFData((CFDataRef)data); //toll-free ARC bridging
        
        CGImageRef imageRef = CGImageCreate(cols,                        //width
                                           rows,                        //height
                                           8,                           //bits per component
                                           8 * 4,                       //bits per pixel
                                           cols * 4,                    //bytes per row
                                           colorSpace,                  //Quartz color space
                                           bitmapInfo,                  //Bitmap info (alpha channel?, order, etc)
                                           provider,                    //Source of data for bitmap
                                           NULL,                        //decode
                                           false,                       //pixel interpolation
                                           kCGRenderingIntentDefault);  //rendering intent
            
        
        _depthImageView.image = [UIImage imageWithCGImage:imageRef];
        
        CGImageRelease(imageRef);
        CGDataProviderRelease(provider);
        CGColorSpaceRelease(colorSpace);
    }
    
}

- (void) renderNormalsFrame: (STDepthFrame*) depthFrame
{
    if(recording) {
        
    } else {

    // Convert depth units from shift to millimeters (stored as floats)
    [_floatDepthFrame updateFromDepthFrame:depthFrame];
    
    // Estimate surface normal direction from depth float values
    STNormalFrame *normalsFrame = [_normalsEstimator calculateNormalsWithDepthFrame:_floatDepthFrame];
    
    size_t cols = normalsFrame.width;
    size_t rows = normalsFrame.height;
    
    // Convert normal unit vectors (ranging from -1 to 1) to RGB (ranging from 0 to 255)
    // Z can be slightly positive in some cases too!
    if (_normalsBuffer == NULL)
    {
        _normalsBuffer = (uint8_t*)malloc(cols * rows * 4);
    }
    for (size_t i = 0; i < cols * rows; i++)
    {
        _normalsBuffer[4*i+0] = (uint8_t)( ( ( normalsFrame.normals[i].x / 2 ) + 0.5 ) * 255);
        _normalsBuffer[4*i+1] = (uint8_t)( ( ( normalsFrame.normals[i].y / 2 ) + 0.5 ) * 255);
        _normalsBuffer[4*i+2] = (uint8_t)( ( ( normalsFrame.normals[i].z / 2 ) + 0.5 ) * 255);
    }
    
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    
    CGBitmapInfo bitmapInfo;
    bitmapInfo = (CGBitmapInfo)kCGImageAlphaNoneSkipFirst;
    bitmapInfo |= kCGBitmapByteOrder32Little;
    
    NSData *data = [NSData dataWithBytes:_normalsBuffer length:cols * rows * 4];
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((CFDataRef)data);
    
    CGImageRef imageRef = CGImageCreate(cols,
                                        rows,
                                        8,
                                        8 * 4,
                                        cols * 4,
                                        colorSpace,
                                        bitmapInfo,
                                        provider,
                                        NULL,
                                        false,
                                        kCGRenderingIntentDefault);
    
    _normalsImageView.image = [[UIImage alloc] initWithCGImage:imageRef];
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    }

}

- (void)renderColorFrame:(CMSampleBufferRef)sampleBuffer
{

    if(!sequence){
        CVImageBufferRef pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    CVPixelBufferLockBaseAddress(pixelBuffer, 0);
    
    size_t cols = CVPixelBufferGetWidth(pixelBuffer);
    size_t rows = CVPixelBufferGetHeight(pixelBuffer);
    
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    
    unsigned char *ptr = (unsigned char *) CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0);
    
    NSData *data = [[NSData alloc] initWithBytes:ptr length:rows*cols*4];
    CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
    
    CGBitmapInfo bitmapInfo;
    bitmapInfo = (CGBitmapInfo)kCGImageAlphaNoneSkipFirst;
    bitmapInfo |= kCGBitmapByteOrder32Little;
    
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((CFDataRef)data);
    
    CGImageRef imageRef = CGImageCreate(cols,
                                        rows,
                                        8,
                                        8 * 4,
                                        cols*4,
                                        colorSpace,
                                        bitmapInfo,
                                        provider,
                                        NULL,
                                        false,
                                        kCGRenderingIntentDefault);
    
    UIImage * image = [[UIImage alloc] initWithCGImage:imageRef];
    
    if(useUndist) {
        image = [self undistRGB:image];
    }
    
    _colorImageView.image = image;
    
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    }
    
}

#pragma mark -  AVFoundation

- (BOOL)queryCameraAuthorizationStatusAndNotifyUserIfNotGranted
{
    // This API was introduced in iOS 7, but in iOS 8 it's actually enforced.
    if ([AVCaptureDevice respondsToSelector:@selector(authorizationStatusForMediaType:)])
    {
        AVAuthorizationStatus authStatus = [AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeVideo];
        
        if (authStatus != AVAuthorizationStatusAuthorized)
        {
            NSLog(@"Not authorized to use the camera!");
            
            [AVCaptureDevice requestAccessForMediaType:AVMediaTypeVideo
                                     completionHandler:^(BOOL granted)
             {
                 // This block fires on a separate thread, so we need to ensure any actions here
                 // are sent to the right place.
                 
                 // If the request is granted, let's try again to start an AVFoundation session. Otherwise, alert
                 // the user that things won't go well.
                 if (granted)
                 {
                     
                     dispatch_async(dispatch_get_main_queue(), ^(void) {
                         
                         [self startColorCamera];
                         
                         _appStatus.colorCameraIsAuthorized = true;
                         [self updateAppStatusMessage];
                         
                     });
                     
                 }
                 
             }];
            
            return false;
        }
        
    }
    
    return true;
    
}

- (void)setupColorCamera
{
    // If already setup, skip it
    if (_avCaptureSession)
        return;
    
    bool cameraAccessAuthorized = [self queryCameraAuthorizationStatusAndNotifyUserIfNotGranted];
    
    if (!cameraAccessAuthorized)
    {
        _appStatus.colorCameraIsAuthorized = false;
        [self updateAppStatusMessage];
        return;
    }
    
    // Use VGA color.
    NSString *sessionPreset = AVCaptureSessionPreset640x480;
    
    // Set up Capture Session.
    _avCaptureSession = [[AVCaptureSession alloc] init];
    [_avCaptureSession beginConfiguration];
    
    // Set preset session size.
    [_avCaptureSession setSessionPreset:sessionPreset];
    
    // Create a video device and input from that Device.  Add the input to the capture session.
    _videoDevice = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
    if (_videoDevice == nil)
        assert(0);
    
    // Configure Focus, Exposure, and White Balance
    NSError *error;
    
    // iOS8 supports manual focus at near-infinity, but iOS7 doesn't.
#if __IPHONE_OS_VERSION_MAX_ALLOWED >= 80000
    _avCaptureSupportsFocusNearInfinity = [_videoDevice respondsToSelector:@selector(setFocusModeLockedWithLensPosition:completionHandler:)];
#else
    _avCaptureSupportsFocusNearInfinity = false;
#endif
    
    // Use auto-exposure, and auto-white balance and set the focus to infinity.
    if([_videoDevice lockForConfiguration:&error])
    {
        [_videoDevice setExposureMode:AVCaptureExposureModeContinuousAutoExposure];
        [_videoDevice setWhiteBalanceMode:AVCaptureWhiteBalanceModeContinuousAutoWhiteBalance];
        
        if (_avCaptureSupportsFocusNearInfinity)
        {
            // Set focus at the maximum position allowable (e.g. "near-infinity") to get the
            // best color/depth alignment.
#if __IPHONE_OS_VERSION_MAX_ALLOWED >= 80000
            [_videoDevice setFocusModeLockedWithLensPosition:1.0f completionHandler:nil];
#endif
        }
        else
        {
            [_videoDevice setAutoFocusRangeRestriction:AVCaptureAutoFocusRangeRestrictionFar];
            [_videoDevice setFocusMode:AVCaptureFocusModeContinuousAutoFocus];
        }
        
        [_videoDevice unlockForConfiguration];
    }
    
    //  Add the device to the session.
    AVCaptureDeviceInput *input = [AVCaptureDeviceInput deviceInputWithDevice:_videoDevice error:&error];
    if (error)
    {
        NSLog(@"Cannot initialize AVCaptureDeviceInput");
        assert(0);
    }
    
    [_avCaptureSession addInput:input]; // After this point, captureSession captureOptions are filled.
    
    //  Create the output for the capture session.
    AVCaptureVideoDataOutput* dataOutput = [[AVCaptureVideoDataOutput alloc] init];
    
    // We don't want to process late frames.
    [dataOutput setAlwaysDiscardsLateVideoFrames:YES];
    
    // Use BGRA pixel format.
    [dataOutput setVideoSettings:[NSDictionary
                                  dictionaryWithObject:[NSNumber numberWithInt:kCVPixelFormatType_32BGRA]
                                  forKey:(id)kCVPixelBufferPixelFormatTypeKey]];
    
    // Set dispatch to be on the main thread so OpenGL can do things with the data
    [dataOutput setSampleBufferDelegate:self queue:dispatch_get_main_queue()];
    
    [_avCaptureSession addOutput:dataOutput];
    
    // Force the framerate to 30 FPS, to be in sync with Structure Sensor.
    if ([_videoDevice respondsToSelector:@selector(setActiveVideoMaxFrameDuration:)]
        && [_videoDevice respondsToSelector:@selector(setActiveVideoMinFrameDuration:)])
    {
        // Available since iOS 7.
        if([_videoDevice lockForConfiguration:&error])
        {
            [_videoDevice setActiveVideoMaxFrameDuration:CMTimeMake(1, 30)];
            [_videoDevice setActiveVideoMinFrameDuration:CMTimeMake(1, 30)];
            [_videoDevice unlockForConfiguration];
        }
    }
    else
    {
        NSLog(@"iOS 7 or higher is required. Camera not properly configured.");
        return;
    }
    
    [_avCaptureSession commitConfiguration];
}

- (void)startColorCamera
{
    if (_avCaptureSession && [_avCaptureSession isRunning])
        return;
    
    // Re-setup so focus is lock even when back from background
    if (_avCaptureSession == nil)
        [self setupColorCamera];
    
    // Start streaming color images.
    [_avCaptureSession startRunning];
}

- (void)stopColorCamera
{
    if ([_avCaptureSession isRunning])
    {
        // Stop the session
        [_avCaptureSession stopRunning];
    }
    
    _avCaptureSession = nil;
    _videoDevice = nil;
}

@end
