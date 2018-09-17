package org.springframework.cloud.stream.app.face.detection.mtcnn.processor.attic;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.io.IOException;
import java.io.InputStream;

import javax.imageio.ImageIO;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgproc;
import org.datavec.image.loader.Java2DNativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.util.ArrayUtil;

import org.springframework.core.io.DefaultResourceLoader;

import static java.awt.image.BufferedImage.TYPE_INT_BGR;
import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

/**
 * @author Christian Tzolov
 */
public class TestImageScale {

	private final static int[] COLOR_CHANNELS = new int[] { 0, 1, 2 };

	private static int COLOR_TYPE = TYPE_INT_BGR;

	private static Java2DNativeImageLoader imageLoader = new Java2DNativeImageLoader();

	public static void main(String[] args) throws IOException {
		// Buffered Image
		BufferedImage img = ImageIO.read(new DefaultResourceLoader().getResource("classpath:/ivan.jpg").getInputStream());

		INDArray ndImageA = Nd4j.create(imageToFloatArray(toIntBufferedImage(img)), new long[] { img.getHeight(), img.getWidth(), 3 });

		BufferedImage imgScaled = scaleBufferedImage(img, 337, 337);

		INDArray ndImageAScaled = Nd4j.create(imageToFloatArray(toIntBufferedImage(imgScaled)), new long[] { imgScaled.getHeight(), imgScaled.getWidth(), 3 });


		// DataVec


		InputStream imgInputStream = new DefaultResourceLoader().getResource("classpath:/ivan.jpg").getInputStream();
		// BGR (Default)
		INDArray ndImageB_BGR_loaded = imageLoader.asMatrix(imgInputStream);

		INDArray ndImageB_BGR = ndImageB_BGR_loaded.get(point(0), all(), all(), all()).permute(1, 2, 0).dup();

		// to RGB
		opencv_core.Mat mat = imageLoader.asMat(ndImageB_BGR_loaded);
		opencv_imgproc.cvtColor(mat, mat, opencv_imgproc.CV_BGR2RGB);
		INDArray ndImageB_RGB = imageLoader.asMatrix(mat).get(point(0), all(), all(), all()).permute(1, 2, 0).dup();


		// Scaled
		INDArray AA0 = ndImageA.permute(2, 0, 1).dup();
		INDArray BB0 = ndImageB_BGR.permute(2, 0, 1).dup();

		opencv_core.Mat matBGR = imageLoader.asMat(BB0);
		opencv_core.Size newSize = new opencv_core.Size(337, 337);
		opencv_imgproc.resize(matBGR, matBGR, newSize, 0, 0, opencv_imgproc.CV_INTER_AREA);
		//INDArray ndResizedImage1 = Nd4j.zeros(matBGR.cols() * matBGR.rows() * matBGR.channels())
		//		.reshape(mat.channels(), matBGR.rows(), matBGR.cols());
		INDArray ndImageBScaled = Transforms.round(imageLoader.asMatrix(matBGR).get(point(0), all(), all(), all()).permute(1, 2, 0).dup());

		INDArray ndImageBScaled2 = scaleNDArrayImage2(BB0, 337, 337);
		INDArray ndImageBScaled3 = scaleNDArrayImage(BB0, 337, 337).dup();

		System.out.println("boza");

	}

	private static float[] imageToFloatArray(BufferedImage bi) {
		int[] ia = ((DataBufferInt) bi.getRaster().getDataBuffer()).getData();
		float[] fa = new float[ia.length * 3];
		for (int i = 0; i < ia.length; i++) {
			final int val = ia[i];
			fa[i * 3 + COLOR_CHANNELS[0]] = ((val >> 16) & 0xFF); //R
			fa[i * 3 + COLOR_CHANNELS[1]] = ((val >> 8) & 0xFF);  //G
			fa[i * 3 + COLOR_CHANNELS[2]] = (val & 0xFF);         //B


			int r = ((val >> 16) & 0xFF);
			int g = ((val >> 8) & 0xFF);
			int b = (val & 0xFF);

			Color c = new Color(val);
			int r1 = c.getRed();
			int g1 = c.getGreen();
			int b1 = c.getBlue();
			int val2 = c.getRGB() & 0xFFFFFF;

			int val3 = new Color((val >> 16) & 0xFF, // R
					(val >> 8) & 0xFF,             // G
					(val & 0xFF))// B
					.getRGB() & 0xFFFFFF;
//			System.out.print(".");

		}
		return fa;
	}

	public static BufferedImage toIntBufferedImage(BufferedImage originalImage) {

		int newWidth = originalImage.getWidth();
		int newHeight = originalImage.getHeight();

		BufferedImage resizedImage = new BufferedImage(newWidth, newHeight, COLOR_TYPE);
		Graphics2D g2d = resizedImage.createGraphics();

		g2d.drawImage(originalImage, 0, 0, newWidth, newHeight, null);
		g2d.dispose();

		return resizedImage;
	}

	public static BufferedImage scaleBufferedImage(BufferedImage originalImage, int newWidth, int newHeight) {

		BufferedImage resizedImage = new BufferedImage(newWidth, newHeight, COLOR_TYPE);
		Graphics2D g2d = resizedImage.createGraphics();
		g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
		g2d.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_SPEED);

		g2d.drawImage(originalImage, 0, 0, newWidth, newHeight, null);
		g2d.dispose();

		return resizedImage;
	}

	private static INDArray scaleNDArrayImage(INDArray ndImage, int width, int height) {

		// 1. Convert INDArray augmentation into  BufferedImage. Sum up the B, G, R byte columns into a single INT column.
		// E.g. (B + G * 8 + R * 16) -> int matrix -> int[][] -> flatten -> int[]
		int[] imagePixels = ArrayUtil.flatten(ndImage.get(all(), all(), point(0)) // << 0
						.addi(ndImage.get(all(), all(), point(1)).muli(256)) // << 8
						.addi(ndImage.get(all(), all(), point(2)).muli(65536)) // << 16
						.toIntMatrix());

		int[] shape = ArrayUtil.toInts(ndImage.shape());
		BufferedImage bufferedImage = new BufferedImage(shape[1], shape[0], BufferedImage.TYPE_INT_RGB);
		bufferedImage.setRGB(0, 0, shape[1], shape[0], imagePixels, 0, shape[1]);

		// 2. Scale the buffered augmentation
		BufferedImage scaledBufferedImage = scaleBufferedImage(bufferedImage, width, height);

		// testWriteImage(scaledBufferedImage, "scaleNDArrayImage");

		// 3. Convert the BufferedImage back to float INDArray
		INDArray result = Nd4j.create(imageToFloatArray(scaledBufferedImage), new long[] { height, width, 3 });


		return result;
	}

	private static INDArray scaleNDArrayImage2(INDArray ndImage, int width, int height) throws IOException {

		opencv_core.Mat matBGR = imageLoader.asMat(ndImage);

		opencv_core.Size newSize = new opencv_core.Size(width, height);
		opencv_imgproc.resize(matBGR, matBGR, newSize, 0, 0, opencv_imgproc.CV_INTER_AREA);

		//INDArray ndResizedImage1 = Nd4j.zeros(matBGR.cols() * matBGR.rows() * matBGR.channels())
		//		.reshape(mat.channels(), matBGR.rows(), matBGR.cols());


		INDArray ndImageBScaled = Transforms.round(imageLoader.asMatrix(matBGR).get(point(0), all(), all(), all()).permute(1, 2, 0).dup());


		return ndImageBScaled;
	}
}
