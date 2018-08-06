package org.springframework.cloud.stream.app.semantic.segmentation.processor;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Image;
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

import org.tensorflow.Tensor;
import org.tensorflow.types.UInt8;

import org.springframework.cloud.stream.app.computer.vision.common.segmentation.SemanticSegmentationService;
import org.springframework.cloud.stream.app.tensorflow.processor.TensorFlowService;
import org.springframework.cloud.stream.app.tensorflow.util.GraphicsUtils;
import org.springframework.core.io.DefaultResourceLoader;

import static java.awt.image.BufferedImage.TYPE_3BYTE_BGR;
import static org.springframework.cloud.stream.app.tensorflow.util.GraphicsUtils.toBufferedImage;

/**
 *
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
 * @author Christian Tzolov
 */
public class SemanticSegmentationJava2D implements SemanticSegmentationService<BufferedImage> {

	public static final String INPUT_TENSOR_NAME = "ImageTensor:0";
	public static final String OUTPUT_TENSOR_NAME = "SemanticPredictions:0";

	private static final int BATCH_SIZE = 1;
	private static final long CHANNELS = 3;
	private static final int REQUIRED_INPUT_IMAGE_SIZE = 513;

	@Override
	public BufferedImage scaledImage(String imagePath) {
		try {
			return scaledImage(ImageIO.read(new DefaultResourceLoader().getResource(imagePath).getInputStream()));
		}
		catch (IOException e) {
			throw new IllegalStateException("Failed to load Image from: " + imagePath, e);
		}
	}

	@Override
	public BufferedImage scaledImage(byte[] image) {
		try {
			return scaledImage(ImageIO.read(new ByteArrayInputStream(image)));
		}
		catch (IOException e) {
			throw new IllegalStateException("Failed to load Image from byte array", e);
		}
	}

	@Override
	public BufferedImage scaledImage(BufferedImage image) {
		double scaleRatio = 1.0 * REQUIRED_INPUT_IMAGE_SIZE / Math.max(image.getWidth(), image.getHeight());
		return scale(image, scaleRatio);
	}

	private BufferedImage scale(BufferedImage originalImage, double scale) {
		int newWidth = (int) (originalImage.getWidth() * scale);
		int newHeight = (int) (originalImage.getHeight() * scale);

		Image tmpImage = originalImage.getScaledInstance(newWidth, newHeight, Image.SCALE_DEFAULT);
		//BufferedImage resizedImage = new BufferedImage(newWidth, newHeight, TYPE_INT_BGR);
		BufferedImage resizedImage = new BufferedImage(newWidth, newHeight, TYPE_3BYTE_BGR);
		//BufferedImage resizedImage = new BufferedImage(newWidth, newHeight, originalImage.getType());

		Graphics2D g2d = resizedImage.createGraphics();
		g2d.drawImage(tmpImage, 0, 0, null);
		g2d.dispose();

		return resizedImage;
	}

	@Override
	public BufferedImage blendMask(BufferedImage mask, BufferedImage background) {
		GraphicsUtils.overlayImages(background, mask, 0, 0);
		return background;
	}

	@Override
	public Tensor<UInt8> createInputTensor(BufferedImage scaledImage) {
		if (scaledImage.getType() != TYPE_3BYTE_BGR) {
			throw new IllegalArgumentException(
					String.format("Expected 3-byte BGR encoding in BufferedImage, found %d", scaledImage.getType()));
		}

		// ImageIO.read produces BGR-encoded images, while the model expects RGB.
		byte[] data = bgrToRgb(toBytes(scaledImage));

		// Expand dimensions since the model expects images to have shape: [1, None, None, 3]
		long[] shape = new long[] { BATCH_SIZE, scaledImage.getHeight(), scaledImage.getWidth(), CHANNELS };

		return Tensor.create(UInt8.class, shape, ByteBuffer.wrap(data));
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

	private byte[] toBytes(BufferedImage bufferedImage) {
		return ((DataBufferByte) bufferedImage.getRaster().getDataBuffer()).getData();
	}

	@Override
	public BufferedImage createMaskImage(int[][] maskPixels, int width, int height, double transparency) {

		maskPixels = rotate(maskPixels);

		int maskWidth = maskPixels.length;
		int maskHeight = maskPixels[0].length;

		int[] maskArray = new int[maskWidth * maskHeight];
		int k = 0;
		for (int i = 0; i < maskHeight; i++) {
			for (int j = 0; j < maskWidth; j++) {
				Color c = (maskPixels[j][i] == 0) ? Color.BLACK : GraphicsUtils.getClassColor(maskPixels[j][i]);
				int t = (int) (255 * (1 - transparency));
				maskArray[k++] = new Color(c.getRed(), c.getGreen(), c.getBlue(), t).getRGB();
			}
		}

		// Turn the pixel array into image;
		BufferedImage maskImage = new BufferedImage(maskWidth, maskHeight, BufferedImage.TYPE_INT_ARGB);
		maskImage.setRGB(0, 0, maskWidth, maskHeight, maskArray, 0, maskWidth);

		// Stretch the image to fit the target box width and height!
		return toBufferedImage(maskImage.getScaledInstance(width, height, Image.SCALE_SMOOTH));

	}

	/**
	 * rotate clockwise in 90 degree
	 * @param input The 2D matrix to be rotated
	 * @return The input matrix rotated clockwise in 90 degrees
	 */
	private int[][] rotate(int[][] input) {

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

	public static void main(String[] args) throws IOException {

		// PASCAL VOC 2012
		//String tensorflowModelLocation = "file:/Users/ctzolov/Downloads/deeplabv3_mnv2_pascal_train_aug/frozen_inference_graph.pb";
		//String imagePath = "classpath:/images/VikiMaxiAdi.jpg";

		// CITYSCAPE
		//String tensorflowModelLocation = "file:/Users/ctzolov/Downloads/deeplabv3_mnv2_cityscapes_train/frozen_inference_graph.pb";
		//String imagePath = "classpath:/images/amsterdam-cityscape1.jpg";
		//String imagePath = "classpath:/images/amsterdam-channel.jpg";
		//String imagePath = "classpath:/images/landsmeer.png";

		// ADE20K
		String tensorflowModelLocation = "file:/Users/ctzolov/Downloads/deeplabv3_xception_ade20k_train/frozen_inference_graph.pb";
		String imagePath = "classpath:/images/interior.jpg";

		BufferedImage inputImage = ImageIO.read(new DefaultResourceLoader().getResource(imagePath).getInputStream());

		TensorFlowService tf = new TensorFlowService(new DefaultResourceLoader().getResource(tensorflowModelLocation));

		SemanticSegmentationJava2D segmentationService = new SemanticSegmentationJava2D();

		BufferedImage scaledImage = segmentationService.scaledImage(inputImage);

		Tensor<UInt8> inTensor = segmentationService.createInputTensor(scaledImage);

		Map<String, Tensor<?>> output = tf.evaluate(Collections.singletonMap(INPUT_TENSOR_NAME, inTensor),
				Arrays.asList(OUTPUT_TENSOR_NAME));

		Tensor<?> maskPixelsTensor = output.get(OUTPUT_TENSOR_NAME);

		int height = (int) maskPixelsTensor.shape()[1];
		int width = (int) maskPixelsTensor.shape()[2];
		long[][] maskPixels = maskPixelsTensor.copyTo(new long[BATCH_SIZE][height][width])[0]; // take 0 because the batch size is 1.

		int[][] maskPixelsInt = segmentationService.toIntArray(maskPixels);

		BufferedImage mask = segmentationService.createMaskImage(maskPixelsInt, scaledImage.getWidth(), scaledImage.getHeight(), 0.35);

		BufferedImage blended = segmentationService.blendMask(mask, scaledImage);

		ImageIO.write(mask, "png", new File("./spring-cloud-starter-stream-processor-semantic-segmentation/target/java2Dmask.jpg"));
		ImageIO.write(blended, "png", new File("./spring-cloud-starter-stream-processor-semantic-segmentation/target/java2Dblended.jpg"));
	}
}
