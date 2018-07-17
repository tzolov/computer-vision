package org.springframework.cloud.stream.app.frame.grabber.source;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;

/**
 * @author Christian Tzolov
 */
public class Main {

	final private static int WEBCAM_DEVICE_INDEX = 0;

	final private static int FRAME_RATE = 30;
	final private static int GOP_LENGTH_IN_FRAMES = 60;

	private static long startTime = 0;
	private static long videoTS = 0;

	public static void main(String[] args) throws FrameGrabber.Exception {
		//int captureWidth = 320;
		//int captureHeight = 240;
		//int captureWidth = 1270;
		//int captureHeight = 710;

		int captureWidth = 1280;
		int captureHeight = 720;

		// The available FrameGrabber classes include OpenCVFrameGrabber (opencv_videoio),
		// DC1394FrameGrabber, FlyCaptureFrameGrabber, OpenKinectFrameGrabber,
		// PS3EyeFrameGrabber, VideoInputFrameGrabber, and FFmpegFrameGrabber.
		//OpenCVFrameGrabber grabber = new OpenCVFrameGrabber(WEBCAM_DEVICE_INDEX);
		FrameGrabber grabber = FrameGrabber.createDefault(WEBCAM_DEVICE_INDEX);
		System.out.println(grabber.getClass().getCanonicalName());
		grabber.setImageWidth(captureWidth);
		grabber.setImageHeight(captureHeight);

		System.out.println(grabber.getFrameRate());

		grabber.start();

		// A really nice hardware accelerated component for our preview...
		CanvasFrame cFrame = new CanvasFrame("Capture Preview", CanvasFrame.getDefaultGamma() / grabber.getGamma());
		//cFrame.setCanvasSize(captureWidth, captureHeight);

		Frame capturedFrame = null;


		// While we are capturing...
		while ((capturedFrame = grabber.grabFrame()) != null) {
			System.out.println(capturedFrame.imageWidth + ":" + capturedFrame.imageHeight);
			System.out.println(capturedFrame.getClass().getCanonicalName());
			//opencv_core.IplImage resizeImage = opencv_core.IplImage.create(120, 120, capturedFrame.depth(), origImg.nChannels());
			if (cFrame.isVisible()) {
				// Show our frame in the preview
				cFrame.showImage(capturedFrame);
			}

			// Let's define our start time...
			// This needs to be initialized as close to when we'll use it as
			// possible,
			// as the delta from assignment to computed time could be too high
			if (startTime == 0)
				startTime = System.currentTimeMillis();

			// Create timestamp for this frame
			videoTS = 1000 * (System.currentTimeMillis() - startTime);

			// Check for AV drift
			//if (videoTS > recorder.getTimestamp())
			//{
			//	System.out.println(
			//			"Lip-flap correction: "
			//					+ videoTS + " : "
			//					+ recorder.getTimestamp() + " -> "
			//					+ (videoTS - recorder.getTimestamp()));
			//
			//	// We tell the recorder to write this frame at this timestamp
			//	recorder.setTimestamp(videoTS);
			//}

			//// Send the frame to the org.bytedeco.javacv.FFmpegFrameRecorder
			//recorder.record(capturedFrame);
		}

		cFrame.dispose();
//		recorder.stop();
		grabber.stop();
	}

}
