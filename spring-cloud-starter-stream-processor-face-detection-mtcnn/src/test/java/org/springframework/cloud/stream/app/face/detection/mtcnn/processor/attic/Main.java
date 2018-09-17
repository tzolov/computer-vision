package org.springframework.cloud.stream.app.face.detection.mtcnn.processor.attic;

import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.DataBufferInt;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.imageio.ImageIO;

import org.tensorflow.Tensor;

import org.springframework.cloud.stream.app.tensorflow.processor.TensorFlowService;
import org.springframework.cloud.stream.app.tensorflow.util.GraphicsUtils;
import org.springframework.core.io.DefaultResourceLoader;
import org.springframework.util.Assert;

import static java.awt.image.BufferedImage.TYPE_INT_BGR;

/**
 * @author Christian Tzolov
 */
public class Main {

	private static final long BATCH_SIZE = 1;
	private static final long CHANNELS = 3;
	//private final static int[] COLOR_CHANNELS = new int[] { 0, 1, 2 };
	private final static int[] COLOR_CHANNELS = new int[] { 2, 1, 0 };


	private final TensorFlowService outputNet;
	private final TensorFlowService refineNet;
	private final TensorFlowService proposeNet;

	private int minFaceSize = 20;
	private double scaleFactor = 0.709;
	private double[] stepsThreshold = new double[] { 0.6, 0.7, 0.7 };

	public int getMinFaceSize() {
		return minFaceSize;
	}

	public void setMinFaceSize(int minFaceSize) {
		this.minFaceSize = minFaceSize;
	}

	public double getScaleFactor() {
		return scaleFactor;
	}

	public void setScaleFactor(double scaleFactor) {
		this.scaleFactor = scaleFactor;
	}

	public Main() throws IOException {
		outputNet = new TensorFlowService(new DefaultResourceLoader().getResource("classpath:/model/onet_graph.proto"));
		refineNet = new TensorFlowService(new DefaultResourceLoader().getResource("classpath:/model/rnet_graph.proto"));
		proposeNet = new TensorFlowService(new DefaultResourceLoader().getResource("classpath:/model/pnet_graph.proto"));
	}

	private List<Double> computeScalePyramid(double m, int minLayer) {
		List<Double> scales = new ArrayList<>();

		int factorCount = 0;

		while (minLayer >= 12) {
			scales.add(m * Math.pow(this.getScaleFactor(), factorCount));
			minLayer = (int) (minLayer * this.getScaleFactor());
			factorCount++;
		}

		return scales;
	}

	private BufferedImage scaleImage(BufferedImage originalImage, double scale) {

		int newWidth = (int) Math.ceil(originalImage.getWidth() * scale);
		int newHeight = (int) Math.ceil(originalImage.getHeight() * scale);

		Image tmpImage = originalImage.getScaledInstance(newWidth, newHeight, Image.SCALE_SMOOTH);
		BufferedImage resizedImage = new BufferedImage(newWidth, newHeight, TYPE_INT_BGR);

		Graphics2D g2d = resizedImage.createGraphics();
		g2d.drawImage(tmpImage, 0, 0, null);
		g2d.dispose();

		return resizedImage;
	}

	private Tensor<Float> makeImageTensor(BufferedImage img) {
		//if (augmentation.getType() != BufferedImage.TYPE_3BYTE_BGR) {
		//	throw new IllegalArgumentException(
		//			String.format("Expected 3-byte BGR encoding in BufferedImage, found %d", augmentation.getType()));
		//}

		// ImageIO.read produces BGR-encoded images, while the model expects RGB.

		int[] data = toIntArray(img);

		int[] transposedData = transpose(data, img.getWidth(), img.getHeight());

		float[] rgbData = toRgbFloat(transposedData);

		float[] rgbNormalizedData = normalize(rgbData);

		//Expand dimensions since the model expects images to have shape: [1, None, None, 3]
		long[] shape = new long[] { BATCH_SIZE, img.getHeight(), img.getWidth(), CHANNELS };

		return Tensor.create(shape, FloatBuffer.wrap(rgbNormalizedData));
	}


	private byte[] bgrToRgb(byte[] brgImage) {
		byte[] rgbImage = new byte[brgImage.length];
		for (int i = 0; i < brgImage.length; i += 3) {
			rgbImage[i] = brgImage[i + 2];
			rgbImage[i + 1] = brgImage[i + 1];
			rgbImage[i + 2] = brgImage[i];
		}
		return rgbImage;
	}

	private static byte[] toBytes(BufferedImage bufferedImage) {
		return ((DataBufferByte) bufferedImage.getRaster().getDataBuffer()).getData();
	}

	private int[] toIntArray(BufferedImage image) {
		BufferedImage imgToRecognition = GraphicsUtils.toBufferedImage(image);
		return ((DataBufferInt) imgToRecognition.getRaster().getDataBuffer()).getData();
	}

	/**
	 * Convert every Int value into a R, G, B bytes triple
	 * @param data
	 * @return
	 */
	private float[] toRgbFloat(int[] data) {
		float[] float_image = new float[data.length * 3];
		for (int i = 0; i < data.length; ++i) {
			final int val = data[i];
			float_image[i * 3 + COLOR_CHANNELS[0]] = ((val >> 16) & 0xFF); //R
			float_image[i * 3 + COLOR_CHANNELS[1]] = ((val >> 8) & 0xFF);  //G
			float_image[i * 3 + COLOR_CHANNELS[2]] = (val & 0xFF);         //B
		}
		return float_image;
	}

	/**
	 * Converts the [0, 255] value range into [-1, +1]
	 * @param notNormalized array of values in the range [0, 255]
	 * @return Returns same values converted to the [-1, +1] range
	 */
	private float[] normalize(float[] notNormalized) {
		float[] normalized = new float[notNormalized.length];
		for (int i = 0; i < notNormalized.length; i++) {
			Assert.isTrue(notNormalized[i] >= 0 && notNormalized[i] <= 255, "Incorrect input rage:" + notNormalized[i]);

			normalized[i] = (notNormalized[i] - 127.5f) * 0.0078125f;
		}
		return normalized;
	}


	/**
	 * Transposes the X and Y axes of the input matrix (e.g. equal to rotate clockwise in 90 degree)
	 * @param input The 2D matrix to be transposed
	 * @return
	 */
	private static int[][] transpose(int[][] input) {

		int w = input.length;
		int h = input[0].length;

		int[][] output = new int[h][w];
		for (int y = 0; y < h; y++) {
			for (int x = w - 1; x >= 0; x--) {
				output[y][x] = input[x][y];
			}
		}
		return output;
	}

	private static float[][][][] transpose4d(float[][][][] input) {

		int b = input.length;
		int w = input[0].length;
		int h = input[0][0].length;
		int c = input[0][0][0].length;

		float[][][][] output = new float[b][h][w][c];
		for (int bi = 0; bi < b; bi++) {
			for (int y = 0; y < h; y++) {
				for (int x = w - 1; x >= 0; x--) {
					for (int ci = 0; ci < c; ci++) {
						output[bi][y][x][ci] = input[bi][x][y][ci];
					}
				}
			}
		}
		return output;
	}

	private static int[] transpose(int[] input, int w, int h) {

		Assert.isTrue(input.length == h * w, "Mismatch");
		int[] output = new int[h * w];
		for (int y = h - 1; y >= 0; y--) {
			for (int x = 0; x < w; x++) {
				output[y + x * h] = input[x + y * w];
			}
		}
		return output;
	}

	public static void main(String[] args) throws IOException {


		Main main = new Main();

		BufferedImage img = ImageIO.read(
				new DefaultResourceLoader().getResource("classpath:/ivan.jpg").getInputStream());

		int height = img.getHeight();
		int width = img.getWidth();

		//scale pyramid
		double m = (double) 12 / main.getMinFaceSize();
		int minLayer = (int) (Math.min(height, width) * m);

		List<Double> scales = main.computeScalePyramid(m, minLayer);

		System.out.println("Scales:" + scales);
		System.out.println("Scales Size:" + scales.size());

		main.stageOne(img, scales);
	}

	private void generateBoundingBox(float[][][] imap, float[][][] reg, double scale, double stepThreshold) {

		System.out.println("boza");
	}

	private void stageOne(BufferedImage image, List<Double> scales) {
		for (Double scale : scales) {

			BufferedImage scaledImage = this.scaleImage(image, scale);
			Tensor<Float> inputTensor = makeImageTensor(scaledImage);

			System.out.println(inputTensor);
			HashMap<String, Object> input = new HashMap<>();
			input.put("pnet/input:0", inputTensor);
			List<String> outputName = new ArrayList<>();
			outputName.add("pnet/conv4-2/BiasAdd");
			outputName.add("pnet/prob1");
			Map<String, Tensor<?>> output = this.proposeNet.evaluate(input, outputName);

			System.out.println(output);

			Tensor<Float> out0 = (Tensor<Float>) output.get("pnet/conv4-2/BiasAdd");
			float[][][][] tensorData0 = out0.copyTo(new float[(int) out0.shape()[0]]
					[(int) out0.shape()[1]][(int) out0.shape()[2]][(int) out0.shape()[3]]);
			float[][][][] tensorData0T = transpose4d(tensorData0);


			Tensor<Float> out1 = (Tensor<Float>) output.get("pnet/prob1");
			float[][][][] tensorData1 = out1.copyTo(new float[(int) out1.shape()[0]]
					[(int) out1.shape()[1]][(int) out1.shape()[2]][(int) out1.shape()[3]]);
			float[][][][] tensorData1T = transpose4d(tensorData1);

			System.out.println("Out0: " + scale + ", " + minMax(tensorData0T));
			System.out.println("Out1: " + scale + ", " + minMax(tensorData1T));

			generateBoundingBox(tensorData1T[0], tensorData0T[0], scale, this.stepsThreshold[0]);

		}

		System.out.println("Done");
	}

	private static String minMax(float[][][][] input) {

		float min = Float.MAX_VALUE;
		float max = Float.MIN_VALUE;

		int b = input.length;
		int w = input[0].length;
		int h = input[0][0].length;
		int c = input[0][0][0].length;

		float[][][][] output = new float[b][h][w][c];
		for (int bi = 0; bi < b; bi++) {
			for (int y = 0; y < h; y++) {
				for (int x = w - 1; x >= 0; x--) {
					for (int ci = 0; ci < c; ci++) {
						min = Math.min(min, input[bi][x][y][ci]);
						max = Math.max(max, input[bi][x][y][ci]);
					}
				}
			}
		}
		return String.format("Min: %s, Max: %s", min, max);
	}


}
