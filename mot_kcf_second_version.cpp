#include <iostream>  
#include <string> 
#include "opencv2/opencv.hpp" 
#include <opencv2/opencv.hpp> 
#include <opencv2/video/video.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <math.h>
#include <time.h>
#include "tracker.hpp"
#include<opencv2/imgcodecs/legacy/constants_c.h>
#include<opencv2/imgproc/imgproc_c.h>
using namespace cv;
using namespace std;

int main(int argc, char **argv){
	long long facehandle = 0;
	clock_t startTime, endTime;
	Mat frame;
	//vector <Frameresult> result;
	vector<Rect2d> bboxvec; 
	//Parameters
	int Merge_threshold = 2000;     // square     example, 20^2=400.
	int UnmatchedDuration = 8;		//丢失此帧数后不对其进行匹配
	int minTrackLength = 12;		//最小追踪长度；
	long currentFrame = 1;			//当前帧
	MKCFTracker trackeur(Merge_threshold, UnmatchedDuration, minTrackLength);	
	bool stop=false;
 ///your video or use your camera
	//cv::VideoCapture capture(0);
	cv::VideoCapture capture("video//demo7.mp4");
	//capture.set(CAP_PROP_POS_FRAMES, 110);
	if (!capture.isOpened())
	{
		cout << "load video fails." << endl;
		return -1;
	}
	//保存视频的路径
	string outputVideoPath = "video//test.mp4";
	//获取当前摄像头的视频信息
	cv::Size sWH = cv::Size((int)capture.get(CAP_PROP_FRAME_WIDTH), (int)capture.get(CAP_PROP_FRAME_HEIGHT));
	VideoWriter outputVideo;
	outputVideo.open(outputVideoPath, outputVideo.fourcc('D', 'I', 'V', 'X'), 21.0, sWH);
	double rate = capture.get(CAP_PROP_FPS);
	int delay =(int)(1000 / rate);
	while (!stop)
	{
		bboxvec.clear();
		if (!capture.read(frame))
		{
			cout << "  Cannot read video.  " << endl;
		}
		float a = 1.512f;
		int b =cvRound(a);
		outputVideo.write(frame);
		startTime = clock();
		std::list<Rect2d>rectlist;

///你的人脸检测模型接口，rectlist保存你人脸检测模型接口x,y,w,h

		result = trackeur.track(frame, rectlist, currentFrame);
		endTime = clock();
		double totaltime;
		totaltime = (double)(endTime - startTime) / CLOCKS_PER_SEC;
		cout << "total_time=" << totaltime*1000 << "ms" << endl;
		for (int i = 0; i<result.size(); i++)
		{
			rectangle(frame, result[i].bboxes.tl(), result[i].bboxes.br(), Scalar(100, 255, 0), 2, 8, 0);
			putText(frame, to_string(result[i].label), cvPoint((int)(result[i].bboxes.x), (int)(result[i].bboxes.y + result[i].bboxes.height*0.5)), CV_FONT_HERSHEY_PLAIN,4, Scalar(0, 0, 255),4,8);
		}
	
		namedWindow("Original video", 2);
		imshow("Original video", frame);
		//Esc to quit.
		int c = cv::waitKey(delay);

		//if ((char)c == 27 || currentFrame >= frameToStop){
		if ((char)c == 27) {
			stop = true;
		}
		//Esc to quit.
		//currentFrame++;
	}
	outputVideo.release();
	std::system("pause");
	return 1;
}//main
