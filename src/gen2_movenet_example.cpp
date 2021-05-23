#include <iostream>
#include <cstdio>

#include "utility.hpp"

// Inludes common necessary includes for development using depthai library
#include "depthai/depthai.hpp"

static bool syncNN = false;
bool useCamera = true;
std::string imagePath;

int pad = 192;

dai::Pipeline createNNPipeline(std::string nnPath){

    dai::Pipeline p;

    auto colorCam = p.create<dai::node::ColorCamera>();
    auto xlinkOut = p.create<dai::node::XLinkOut>();
    auto nn1 = p.create<dai::node::NeuralNetwork>();
    auto nnOut = p.create<dai::node::XLinkOut>();

    auto xin = p.create<dai::node::XLinkIn>();
    xin->setStreamName("nn_in");


    nn1->setBlobPath(nnPath);

    xlinkOut->setStreamName("preview");
    nnOut->setStreamName("detections");    

    colorCam->setPreviewSize(pad, pad);
    colorCam->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1080_P);
    colorCam->setInterleaved(true);
    colorCam->setColorOrder(dai::ColorCameraProperties::ColorOrder::BGR);
    
    if (useCamera) colorCam->preview.link(nn1->input);
    else xin->out.link(nn1->input);


    if (syncNN) nn1->passthrough.link(xlinkOut->input);
    else colorCam->preview.link(xlinkOut->input);

    nn1->out.link(nnOut->input);

    return p;

}

int main(int argc, char** argv){
    using namespace std;

    std::string nnPath;

    if(argc > 1){
        if (strcmp(argv[1],"-i") == 0)
        {
            if (strcmp(argv[2],"camera") != 0)
            {
                useCamera = false;
                imagePath = argv[2];
            }

            if (strcmp(argv[3],"-m") == 0)
            {
                if (strcmp(argv[4],"lightning")==0) 
                {
                    nnPath = "../src/models/movenet_singlepose_lightning_3.blob";
                }
                else 
                {
                    nnPath = "../src/models/movenet_singlepose_thunder_3.blob";
                    pad = 256;
                }
            }
        }
        else
        {
            printf("Usage: movenet -i [camera|image path] -m [lightning|thunder]\n");
            exit(0);
        }
    }
    else
    {
        printf("Usage: movenet -i [camera|image path] -m [lightning|thunder]\n");
        exit(0);
    }

    // Print which blob we are using
    printf("Using blob at path: %s\n", nnPath.c_str());

    // Create pipeline
    dai::Pipeline p = createNNPipeline(nnPath);

    // Connect to device with above created pipeline
    dai::Device d(p);
    // Start the pipeline
    d.startPipeline();

    cv::Mat frame_orig, frame;
    cv::Mat frame2,frame3;

    auto preview = d.getOutputQueue("preview");
    auto detections = d.getOutputQueue("detections");
    
    
    auto in = d.getInputQueue("nn_in");
    
    int LINES_BODY[16][2] = {{4,2},{2,0},{0,1},{1,3},
                {10,8},{8,6},{6,5},{5,7},{7,9},
                {6,12},{12,11},{11,5},
                {12,14},{14,16},{11,13},{13,15}};
    while(1){

        if (useCamera)
        {
            auto imgFrame = preview->get<dai::ImgFrame>();
            if(imgFrame){
                printf("Frame - w: %d, h: %d\n", imgFrame->getWidth(), imgFrame->getHeight());
                frame = cv::Mat(imgFrame->getHeight(), imgFrame->getWidth(), CV_8UC3, imgFrame->getData().data());
            }
        }
        else
        {

            frame_orig = cv::imread(imagePath);
            cv::resize(frame_orig,frame, cv::Size(656,656),cv::INTER_AREA);

            cv::resize(frame,frame2, cv::Size(pad,pad),cv::INTER_AREA);
            cv::cvtColor(frame2, frame3, cv::ColorConversionCodes::COLOR_RGB2BGR);
            
            dai::ImgFrame imgFrame;

            imgFrame.setWidth(pad);
            imgFrame.setHeight(pad);
            imgFrame.setType(dai::ImgFrame::Type::RGB888i);
            
            imgFrame.setFrame(frame3);

            in->send(imgFrame);
            
        }

        auto det = detections->get<dai::NNData>();

        std::vector<float> detData = det->getLayerFp16("Identity");

        int landmarks_y[17]; 
        int landmarks_x[17];
        float scores[17];

        if(detData.size() > 0){
            int pos = 0;

            int frameSize;
            if (useCamera) frameSize = pad;
            else frameSize = 656;

            for (int i=0; i<detData.size(); i+=3)
            {
                landmarks_y[pos] = (int) (detData[i] * frameSize);
                landmarks_x[pos] = (int) (detData[i+1] * frameSize);
                scores[pos] = detData[i+2];
                pos++;
            }

            for (int i=0; i<16; i++)
            {
                cv::Point point1 = cv::Point(landmarks_x[LINES_BODY[i][0]],landmarks_y[LINES_BODY[i][0]]);
                cv::Point point2 = cv::Point(landmarks_x[LINES_BODY[i][1]],landmarks_y[LINES_BODY[i][1]]);
                cv::line(frame, point1, point2 ,cv::Scalar(255,255,255));
                cv::circle(frame, point1, 4, cv::Scalar(255,0,0), -11);
                cv::circle(frame, point2, 4, cv::Scalar(255,0,0), -11);
            }
        }
        cv::imshow("preview", frame);
        int key = cv::waitKey(1);
        if (key == 'q'){
            return 0;
        } 
    }

    return 0;
}


    