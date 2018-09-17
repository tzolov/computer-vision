package org.springframework.cloud.stream.app.face.recognition.processor;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.Arrays;

import javax.imageio.ImageIO;

import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.core.CropAndResize;
import org.tensorflow.op.core.ReduceMean;

import org.springframework.cloud.stream.app.tensorflow.util.GraphicsUtils;
import org.springframework.core.io.DefaultResourceLoader;

/**
 * @author Christian Tzolov
 */
public class Main3 {
	public static void main(String[] args) throws IOException {

		try (Graph g = new Graph()) {

			Ops ops = Ops.create(g);


			final String imageUri = "file:/Users/ctzolov/Dev/projects/tf/facenet/data/images/Anthony_Hopkins_0001.jpg";
			//final String imageUri = "file:/Users/ctzolov/Dev/projects/scdf/computer-vision/spring-cloud-starter-stream-processor-face-detection-mtcnn/target/cropped0.png";
			BufferedImage img = ImageIO.read(new DefaultResourceLoader().getResource(imageUri).getInputStream());
			long[] shape = new long[] { BATCH_SIZE, img.getHeight(), img.getWidth(), CHANNELS };
			float[] rgbData = toRgbFloat(toIntArray(img));


			Constant<Float> imgTensor = ops.constant(shape, FloatBuffer.wrap(rgbData));
			Constant<Float> boxes = ops.constant(new float[][] { { 51.0f / img.getHeight(), 47.0f / img.getWidth()
					, 193.0f / img.getHeight(), 215.0f / img.getWidth() } }, Float.class);
			Constant<Integer> boxInd = ops.constant(new int[] { 0 }, Integer.class);
			Constant<Integer> cropSize = ops.constant(new int[] { 160, 160 }, Integer.class);

			CropAndResize cr = ops.cropAndResize(imgTensor, boxes, boxInd, cropSize);
			//ops.cast()



			try (Session s = new Session(g)) {
//				new TensorFlow().
				Tensor<Float> d = s.runner().fetch(cr.asOutput().op().name()).run().get(0).expect(Float.class);
				System.out.println(Arrays.toString(d.shape()));
				//int[][][][] result = new int[1][160][160][3];
				float[][][] result = d.copyTo(new float[1][(int) d.shape()[1]][(int) d.shape()[2]][3])[0];
				System.out.println(Arrays.deepToString(result));

			}
		}
	}

	public static int[] toIntArray(BufferedImage image) {
		BufferedImage imgToRecognition = GraphicsUtils.toBufferedImage(image);
		return ((DataBufferInt) imgToRecognition.getRaster().getDataBuffer()).getData();
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

	public static Operand<Float> preWhiten(Ops ops, Operand<Integer> image) {

		Operand<Float> fImage = ops.cast(image, Float.class);

		Operand<Float> mean = ops.mean(fImage, ops.constant(new long[] { 0, 1, 3 }, Long.class));

		Operand<Float> rm = ops.reduceMean(fImage, ops.constant(new long[] { 0, 1, 3 }, Long.class), ReduceMean.keepDims(true));
		Operand<Float> var = ops.reduceMean(ops.square(ops.sub(fImage, rm)), ops.constant(new long[] { 0, 1, 3 }, Long.class));
		Operand<Float> std = ops.sqrt(var);

//		ops.max(std, )
//		INDArray stdAdj = Transforms.max(std, 1.0 / Math.sqrt(image.length()));
//		INDArray y = image.sub(mean).mul(stdAdj.rdiv(1));
////		INDArray y1 = y.permutei(0, 2, 3, 1).dup();
		return null;
	}
	private static final int BATCH_SIZE = 1;
	private static final long CHANNELS = 3;
	//private final static int[] COLOR_CHANNELS = new int[] { 0, 1, 2 };
	private final static int[] COLOR_CHANNELS = new int[] { 2, 1, 0 };

}
