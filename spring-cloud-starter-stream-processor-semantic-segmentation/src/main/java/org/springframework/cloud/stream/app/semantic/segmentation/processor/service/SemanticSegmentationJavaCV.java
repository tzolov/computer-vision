package org.springframework.cloud.stream.app.semantic.segmentation.processor.service;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Collections;
import java.util.Map;

import javax.imageio.ImageIO;

import org.bytedeco.javacpp.indexer.UByteIndexer;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacv.Java2DFrameUtils;
import org.tensorflow.Tensor;
import org.tensorflow.types.UInt8;

import org.springframework.cloud.stream.app.tensorflow.processor.TensorFlowService;
import org.springframework.cloud.stream.app.tensorflow.util.GraphicsUtils;
import org.springframework.core.io.DefaultResourceLoader;
import org.springframework.util.Assert;

import static org.bytedeco.javacpp.opencv_core.CV_8UC;
import static org.bytedeco.javacpp.opencv_imgproc.CV_BGR2RGB;
import static org.bytedeco.javacpp.opencv_imgproc.cvResize;
import static org.bytedeco.javacpp.opencv_imgproc.cvtColor;

/**
 * Semantic image segmentation - the task of assigning a semantic label, such as “road”, “sky”, “person”, “dog”, to
 * every pixel in an image.
 *
 * https://ai.googleblog.com/2018/03/semantic-image-segmentation-with.html
 * https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md
 * https://github.com/tensorflow/models/tree/master/research/deeplab
 * https://github.com/tensorflow/models/blob/master/research/deeplab/deeplab_demo.ipynb
 * http://presentations.cocodataset.org/Places17-GMRI.pdf
 *
 * http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html
 * https://www.cityscapes-dataset.com/dataset-overview/#class-definitions
 * http://groups.csail.mit.edu/vision/datasets/ADE20K/
 *
 * https://github.com/mapillary/inplace_abn
 *
 * JavaCV/OpenCV
 * https://github.com/bytedeco/javacv/issues/70
 * http://bytedeco.org/
 * https://github.com/bytedeco/javacv
 * https://github.com/bytedeco/javacv/issues/888#issuecomment-361126796
 * https://www.learnopencv.com/alpha-blending-using-opencv-cpp-python/
 * https://medium.com/@alexkn15/tensorflow-and-javacv-591c1b9443a3
 *
 * @author Christian Tzolov
 */
public class SemanticSegmentationJavaCV implements SemanticSegmentationService<opencv_core.IplImage> {
	public static final String INPUT_TENSOR_NAME = "ImageTensor:0";
	public static final String OUTPUT_TENSOR_NAME = "SemanticPredictions:0";

	private static final int BATCH_SIZE = 1;
	private static final int REQUIRED_INPUT_IMAGE_SIZE = 513;

	@Override
	public opencv_core.IplImage scaledImage(String imagePath) {
		try {
			return scaledImage(ImageIO.read(new DefaultResourceLoader().getResource(imagePath).getInputStream()));
		}
		catch (Exception e) {
			throw new IllegalStateException("", e);
		}
	}

	@Override
	public opencv_core.IplImage scaledImage(byte[] image) {
		try {
			return scaledImage(ImageIO.read(new ByteArrayInputStream(image)));
		}
		catch (IOException e) {
			throw new IllegalStateException("", e);
		}
	}

	@Override
	public opencv_core.IplImage scaledImage(BufferedImage image) {
		opencv_core.IplImage inputImage = Java2DFrameUtils.toIplImage(image);
		double scaleRatio = 1.0 * REQUIRED_INPUT_IMAGE_SIZE / Math.max(inputImage.width(), inputImage.height());
		return scaleImage(inputImage, scaleRatio);
	}

	private opencv_core.IplImage scaleImage(opencv_core.IplImage image, double scale) {
		int newWidth = (int) (image.width() * scale);
		int newHeight = (int) (image.height() * scale);
		// RESIZE
		opencv_core.IplImage resizedImage = opencv_core.IplImage.create(newWidth, newHeight, image.depth(), image.nChannels());
		cvResize(image, resizedImage);
		return resizedImage;
	}

	@Override
	public opencv_core.IplImage blendMask(opencv_core.IplImage maskImage, opencv_core.IplImage backgroundImage) {
		opencv_core.Mat mask = new opencv_core.Mat(maskImage);
		opencv_core.Mat background = new opencv_core.Mat(backgroundImage);

		opencv_core.Mat overlay = new opencv_core.Mat(mask.rows(), mask.cols(), CV_8UC(3));
		final UByteIndexer rgbaIdx = mask.createIndexer();
		final UByteIndexer bgrIdx = background.createIndexer();
		final UByteIndexer dstIdx = overlay.createIndexer();
		final int rows = mask.rows(), cols = mask.cols();

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				float a = rgbaIdx.get(i, j, 3) * (1.0f / 255.0f);
				float b = rgbaIdx.get(i, j, 2) * a + bgrIdx.get(i, j, 0) * (1.0f - a);
				float g = rgbaIdx.get(i, j, 1) * a + bgrIdx.get(i, j, 1) * (1.0f - a);
				float r = rgbaIdx.get(i, j, 0) * a + bgrIdx.get(i, j, 2) * (1.0f - a);
				dstIdx.put(i, j, 0, (byte) b);
				dstIdx.put(i, j, 1, (byte) g);
				dstIdx.put(i, j, 2, (byte) r);
			}
		}

		rgbaIdx.release();
		bgrIdx.release();
		dstIdx.release();

		return new opencv_core.IplImage(overlay);
	}

	@Override
	public Tensor<UInt8> createInputTensor(opencv_core.IplImage resizedImage) {
		// BGR to RGB
		opencv_core.Mat mat = new opencv_core.Mat();

		// Copy to prevent altering the colors of the scaled image
		new opencv_core.Mat(resizedImage).copyTo(mat);

		// Convert from BGR to RGB
		cvtColor(mat, mat, CV_BGR2RGB);

		// To Bytes
		//byte[] imageBytes = toBytes(mat);
		byte[] imageBytes = toBytes(mat);

		// Input Tensor
		long[] shape = new long[] { BATCH_SIZE, mat.rows(), mat.cols(), mat.channels() };
		return Tensor.create(UInt8.class, shape, ByteBuffer.wrap(imageBytes));
	}

	private byte[] toBytes(opencv_core.Mat mat) {
		BufferedImage bufferedImage = Java2DFrameUtils.toBufferedImage(mat);
		return ((DataBufferByte) bufferedImage.getRaster().getDataBuffer()).getData();
	}

	private byte[] toBytes2(opencv_core.Mat image) {

		byte[] bytes = new byte[image.cols() * image.rows() * 3];

		int bi = 0;
		UByteIndexer index = image.createIndexer();
		for (int i = 0; i < image.rows(); i++) {
			for (int j = 0; j < image.cols(); j++) {
				int[] pixel = new int[3];
				index.get(i, j, pixel);

				bytes[bi] = (byte) pixel[0];
				bytes[bi + 1] = (byte) pixel[1];
				bytes[bi + 2] = (byte) pixel[2];

				bi = bi + 3;
			}
		}
		index.release();
		return bytes;
	}

	public int[][] toIntArray(long[][] longArray) {
		int[][] intArray = new int[longArray.length][longArray[0].length];
		for (int i = 0; i < longArray.length; i++) {
			for (int j = 0; j < longArray[0].length; j++) {
				intArray[i][j] = (int) longArray[i][j];
			}
		}
		return intArray;
	}

	@Override
	public opencv_core.IplImage createMaskImage(int[][] maskPixels, int width, int height, double transparency) {
		Assert.isTrue(transparency >= 0.0 && transparency <= 1.0, "transparency accepts values between [0, 1]");
		int maskRows = maskPixels.length;
		int maskCols = maskPixels[0].length;
		int channels = 3;

		opencv_core.Mat mat = new opencv_core.Mat(maskRows, maskCols, CV_8UC(channels + 1));

		// Rote on the fly
		UByteIndexer matIndex = mat.createIndexer();
		for (int j = 0; j < maskCols; j++) {
			for (int i = maskRows - 1; i >= 0; i--) {

				Color c = (maskPixels[i][j] == 0) ? Color.BLACK : GraphicsUtils.getClassColor(maskPixels[i][j]);

				int[] pixel = new int[4];
				pixel[0] = c.getRed();
				pixel[1] = c.getGreen();
				pixel[2] = c.getBlue();
				pixel[3] = (int) (255 * (1 - transparency));
				matIndex.put(i, j, pixel);
			}
		}
		matIndex.release();

		// Rotate -- NOT necessary as we rotated the image above
		//opencv_core.Mat rotatedMat = new opencv_core.Mat();
		//opencv_core.rotate(mat, rotatedMat, ROTATE_90_CLOCKWISE);
		//opencv_core.IplImage img = new opencv_core.IplImage(rotatedMat);

		// Scale
		opencv_core.IplImage img = new opencv_core.IplImage(mat);
		opencv_core.IplImage resizedImage = opencv_core.IplImage.create(width, height, img.depth(), img.nChannels());
		cvResize(img, resizedImage);
		return resizedImage;
	}

	public static void main(String[] args) throws IOException {

		// PASCAL VOC 2012
		String tensorflowModelLocation = "file:/Users/ctzolov/Downloads/deeplabv3_mnv2_pascal_train_aug/frozen_inference_graph.pb";
		String imagePath = "classpath:/images/VikiMaxiAdi.jpg";

		// CITYSCAPE
		// String tensorflowModelLocation = "file:/Users/ctzolov/Downloads/deeplabv3_mnv2_cityscapes_train/frozen_inference_graph.pb";
		//String imagePath = "classpath:/images/amsterdam-cityscape1.jpg";
		//String imagePath = "classpath:/images/amsterdam-channel.jpg";
		// String imagePath = "classpath:/images/landsmeer.png";

		// ADE20K
		//String tensorflowModelLocation = "file:/Users/ctzolov/Downloads/deeplabv3_xception_ade20k_train/frozen_inference_graph.pb";
		//String imagePath = "classpath:/images/interior.jpg";

		TensorFlowService tfService = new TensorFlowService(new DefaultResourceLoader().getResource(tensorflowModelLocation));

		SemanticSegmentationJavaCV segmentationService = new SemanticSegmentationJavaCV();

		opencv_core.IplImage scaledImage = segmentationService.scaledImage(imagePath);

		Tensor<UInt8> inTensor = segmentationService.createInputTensor(scaledImage);

		Map<String, Tensor<?>> outputMap = tfService.evaluate(
				Collections.singletonMap(INPUT_TENSOR_NAME, inTensor),
				Arrays.asList(OUTPUT_TENSOR_NAME));

		Tensor<?> maskPixelsTensor = outputMap.get(OUTPUT_TENSOR_NAME);

		int height = (int) maskPixelsTensor.shape()[1];
		int width = (int) maskPixelsTensor.shape()[2];
		long[][] maskPixels = maskPixelsTensor.copyTo(new long[BATCH_SIZE][height][width])[0]; // take 0 because the batch size is 1.

		int[][] maskPixlesInt = segmentationService.toIntArray(maskPixels);

		opencv_core.IplImage maskImage = segmentationService.createMaskImage(maskPixlesInt, scaledImage.width(), scaledImage.height(), 0.25);

		opencv_core.IplImage blendImage = segmentationService.blendMask(maskImage, scaledImage);

		ImageIO.write(Java2DFrameUtils.toBufferedImage(maskImage), "png", new File("./spring-cloud-starter-stream-processor-semantic-segmentation/target/mask.jpg"));
		ImageIO.write(Java2DFrameUtils.toBufferedImage(scaledImage), "png", new File("./spring-cloud-starter-stream-processor-semantic-segmentation/target/scaled.jpg"));
		ImageIO.write(Java2DFrameUtils.toBufferedImage(blendImage), "png", new File("./spring-cloud-starter-stream-processor-semantic-segmentation/target/blendMask.jpg"));

	}
}
