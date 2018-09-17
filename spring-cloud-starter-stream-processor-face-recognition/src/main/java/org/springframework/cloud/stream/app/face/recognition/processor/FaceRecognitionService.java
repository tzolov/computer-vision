package org.springframework.cloud.stream.app.face.recognition.processor;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import org.tensorflow.Tensor;

import org.springframework.cloud.stream.app.tensorflow.processor.TensorFlowService;
import org.springframework.cloud.stream.app.tensorflow.util.GraphicsUtils;

/**
 * @author Christian Tzolov
 */
public class FaceRecognitionService {

	private static final int BATCH_SIZE = 1;
	private static final long CHANNELS = 3;
	private final static int[] COLOR_CHANNELS = new int[] { 2, 1, 0 };

	private TensorFlowService tfService;

	public FaceRecognitionService(TensorFlowService tfService) {
		this.tfService = tfService;
	}

	public float[] encode(BufferedImage image) {
		Map<String, Object> feeds = new HashMap<>();
		feeds.put("input:0", makeImageTensor(image));
		feeds.put("phase_train:0", Tensor.create(false));

		Map<String, Tensor<?>> output = tfService.evaluate(feeds, Arrays.asList("embeddings:0"));

		Tensor<?> embeddingsTensor = output.get("embeddings:0");


		float[] embeddings = embeddingsTensor.copyTo(new
				float[BATCH_SIZE][(int) embeddingsTensor.shape()[1]])[0]; // take 0 because the batch size is 1.
		return embeddings;
	}

	public static Tensor<Float> makeImageTensor(BufferedImage img) {
		long[] shape = new long[] { BATCH_SIZE, img.getHeight(), img.getWidth(), CHANNELS };
		float[] rgbData = toRgbFloat(toIntArray(img));
		//return Tensor.create(shape, FloatBuffer.wrap(normalize(rgbData)));
		return Tensor.create(shape, FloatBuffer.wrap(rgbData));
	}

	/**
	 * Convert every Int value into a R, G, B bytes triple
	 * @param data
	 * @return
	 */
	public static float[] toRgbFloat(int[] data) {
		float[] float_image = new float[data.length * 3];
		for (int i = 0; i < data.length; ++i) {
			final int val = data[i];
			float_image[i * 3 + COLOR_CHANNELS[0]] = ((val >> 16) & 0xFF); //R
			float_image[i * 3 + COLOR_CHANNELS[1]] = ((val >> 8) & 0xFF);  //G
			float_image[i * 3 + COLOR_CHANNELS[2]] = (val & 0xFF);         //B
		}
		return float_image;
	}


	public static int[] toIntArray(BufferedImage image) {
		BufferedImage imgToRecognition = GraphicsUtils.toBufferedImage(image);
		return ((DataBufferInt) imgToRecognition.getRaster().getDataBuffer()).getData();
	}
}
