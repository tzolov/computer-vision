package org.springframework.cloud.stream.app.face.detection.mtcnn.processor;

import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

import javax.imageio.ImageIO;

import com.sun.javafx.iio.ImageStorage;
import org.tensorflow.Tensor;

import org.springframework.cloud.stream.app.tensorflow.processor.TensorFlowService;
import org.springframework.cloud.stream.app.tensorflow.util.GraphicsUtils;
import org.springframework.core.io.DefaultResourceLoader;

import static java.awt.image.BufferedImage.TYPE_INT_BGR;

/**
 * @author Christian Tzolov
 */
public class Main {

	private final TensorFlowService outputNet;
	private final TensorFlowService refineNet;
	private final TensorFlowService proposeNet;

	private int minFaceSize = 20;
	private double scaleFacor = 0.709;
	private double[] stepsThreshold = new double[] { 0.6, 0.7, 0.7 };

	public Main() throws IOException {
		outputNet = new TensorFlowService(new DefaultResourceLoader().getResource("classpath:/model/onet_graph.proto"));
		refineNet = new TensorFlowService(new DefaultResourceLoader().getResource("classpath:/model/rnet_graph.proto"));
		proposeNet = new TensorFlowService(new DefaultResourceLoader().getResource("classpath:/model/pnet_graph.proto"));
	}

	private List<Double> computeScalePyramid(double m, int minLayer) {
		List<Double> scales = new ArrayList<>();

		int factorCount = 0;

		while (minLayer >= 12) {
			scales.add(m * Math.pow(scaleFacor, factorCount));
			minLayer = (int) (minLayer * scaleFacor);
			factorCount++;
		}

		return scales;
	}

	private BufferedImage scaleImage(BufferedImage originalImage, double scale) {
		int newWidth = (int) (originalImage.getWidth() * scale);
		int newHeight = (int) (originalImage.getHeight() * scale);

		Image tmpImage = originalImage.getScaledInstance(newWidth, newHeight, Image.SCALE_SMOOTH);
		BufferedImage resizedImage = new BufferedImage(newWidth, newHeight, TYPE_INT_BGR);

		Graphics2D g2d = resizedImage.createGraphics();
		g2d.drawImage(tmpImage, 0, 0, null);
		g2d.dispose();

		return resizedImage;
	}

	private void stageOne(BufferedImage image, List<Double> scales) {
		for (Double scale : scales) {
			BufferedImage scaledImage = this.scaleImage(image, scale);

			Tensor<Float> inputTensor = makeImageTensor(scaledImage);

			HashMap<String, Object> input = new HashMap<>();
			input.put("pnet/input:0", inputTensor);
			List<String> outputName = new ArrayList<>();
			outputName.add("pnet/conv4-2/BiasAdd");
			outputName.add("pnet/prob1");
			Map<String, Tensor<?>> output = this.proposeNet.evaluate(input, outputName);

			System.out.println(output);

			Tensor<Float> out0 = (Tensor<Float>) output.get("pnet/conv4-2/BiasAdd");

			float[][][][] tensorData0 = out0.copyTo(new float[1][scaledImage.getHeight()][scaledImage.getHeight()][3]);

			float[][][][] tensorData0T = new float[1][scaledImage.getHeight()][scaledImage.getHeight()][3];

			for (int b = 0; b < 1; b++) {
				for (int z = 0; z < 3; z++) {
					for (int y = 0; y < scaledImage.getHeight(); y++) {
						for (int x = 0; x < scaledImage.getWidth(); x++) {
							tensorData0T[b][x][y][z] = tensorData0[b][y][x][z];
						}
					}
				}
			}


			System.out.println(tensorData0T);
			//
			//img_x = np.expand_dims(scaled_image, 0)
			//img_y = np.transpose(img_x, (0, 2, 1, 3))
			//
			//out = self.__pnet.feed(img_y)
		}

		System.out.println("Done");
	}

	private static final long BATCH_SIZE = 1;
	private static final long CHANNELS = 3;
	private final static int[] COLOR_CHANNELS = new int[] { 0, 1, 2 };

	private Tensor<Float> makeImageTensor(BufferedImage img) {
		//if (img.getType() != BufferedImage.TYPE_3BYTE_BGR) {
		//	throw new IllegalArgumentException(
		//			String.format("Expected 3-byte BGR encoding in BufferedImage, found %d", img.getType()));
		//}

		// ImageIO.read produces BGR-encoded images, while the model expects RGB.
		int[] data = toIntArray(img);

		float[] rgbData = toRgbFloat(data);

		float[] rgbNormalizedData = new float[rgbData.length];

		for (int i = 0; i < data.length - 2; i = i + 3) {
			rgbNormalizedData[i] = (float) ((rgbData[i + 1] - 127.5) * 0.0078125);
			rgbNormalizedData[i + 1] = (float) ((rgbData[i + 2] - 127.5) * 0.0078125);
			rgbNormalizedData[i + 2] = (float) ((rgbData[i] - 127.5) * 0.0078125);
		}
		//Expand dimensions since the model expects images to have shape: [1, None, None, 3]
		long[] shape = new long[] { BATCH_SIZE, img.getHeight(), img.getWidth(), CHANNELS };

		return Tensor.create(shape, FloatBuffer.wrap(rgbNormalizedData));
	}

	private int[] toIntArray(BufferedImage image) {
		BufferedImage imgToRecognition = GraphicsUtils.toBufferedImage(image);
		return ((DataBufferInt) imgToRecognition.getRaster().getDataBuffer()).getData();
	}

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

	public static void main(String[] args) throws IOException {


		Main main = new Main();

		BufferedImage img = ImageIO.read(new DefaultResourceLoader().getResource("classpath:/ivan.jpg").getInputStream());

		int height = img.getHeight();
		int width = img.getWidth();

		//scale pyramid
		double m = (double) 12 / main.minFaceSize;
		int minLayer = (int) (Math.min(height, width) * m);

		List<Double> scales = main.computeScalePyramid(m, minLayer);

		System.out.println("Scales:" + scales);
		System.out.println("Scales Size:" + scales.size());

		main.stageOne(img, scales);
	}
}
