package org.springframework.cloud.stream.app.face.detection.mtcnn.processor.attic;

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import javax.imageio.ImageIO;

import com.fasterxml.jackson.databind.ObjectMapper;
import net.tzolov.cv.mtcnn.FaceAnnotation;
import net.tzolov.cv.mtcnn.MtcnnUtil;
import org.apache.commons.io.IOUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.SpecifiedIndex;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.tensorflow.conversion.graphrunner.GraphRunner;
import org.tensorflow.framework.ConfigProto;


import org.springframework.core.io.DefaultResourceLoader;

import static java.awt.image.BufferedImage.TYPE_INT_BGR;
import static net.tzolov.cv.mtcnn.MtcnnUtil.CHANNEL_COUNT;
import static net.tzolov.cv.mtcnn.MtcnnUtil.C_ORDERING;
import static net.tzolov.cv.mtcnn.MtcnnUtil.NonMaxSuppressionType.Min;
import static net.tzolov.cv.mtcnn.MtcnnUtil.NonMaxSuppressionType.Union;
import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.interval;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

/**
 * @author Christian Tzolov
 */
public class MtcnnServiceJava2D {

	private final static int[] COLOR_CHANNELS = new int[] { 0, 1, 2 };

	//private final TensorFlowService outputNet;
	//private final TensorFlowService refineNet;
	//private final TensorFlowService proposeNet;

	private final GraphRunner proposeNetGraphRunner;
	private final GraphRunner refineNetGraphRunner;
	private final GraphRunner outputNetGraphRunner;

	/**
	 * Minimum faces' size
	 */
	private int minFaceSize = 20;

	/**
	 *
	 */
	private double scaleFactor = 0.709;

	/**
	 *
	 */
	private double[] stepsThreshold = new double[] { 0.6, 0.7, 0.7 };


	public MtcnnServiceJava2D() throws IOException {
		//outputNet = new TensorFlowService(new DefaultResourceLoader().getResource("classpath:/model/onet_graph.proto"));
		//refineNet = new TensorFlowService(new DefaultResourceLoader().getResource("classpath:/model/rnet_graph.proto"));
		//proposeNet = new TensorFlowService(new DefaultResourceLoader().getResource("classpath:/model/pnet_graph.proto"));

		proposeNetGraphRunner = new GraphRunner(IOUtils.toByteArray(new DefaultResourceLoader().getResource("classpath:/model/pnet_graph.proto").getInputStream()),
				Arrays.asList("pnet/input"), ConfigProto.getDefaultInstance());
		refineNetGraphRunner = new GraphRunner(IOUtils.toByteArray(new DefaultResourceLoader().getResource("classpath:/model/rnet_graph.proto").getInputStream()),
				Arrays.asList("rnet/input"), ConfigProto.getDefaultInstance());
		outputNetGraphRunner = new GraphRunner(IOUtils.toByteArray(new DefaultResourceLoader().getResource("classpath:/model/onet_graph.proto").getInputStream()),
				Arrays.asList("onet/input"), ConfigProto.getDefaultInstance());

	}

	/**
	 * Detects faces in an augmentation, and returns bounding boxes and points for them.
	 * @param imageUri Uri of the augmentation to detect
	 * @return Array of face bounding boxes found in the augmentation
	 * @throws IOException Incorrect augmentation Uri.
	 */
	public FaceAnnotation[] faceDetection(String imageUri) throws IOException {
		BufferedImage image = ImageIO.read(new DefaultResourceLoader().getResource(imageUri).getInputStream());
		return this.faceDetection(image);

	}

	/**
	 * Detects faces in an augmentation, and returns bounding boxes and points for them.
	 * @param img augmentation to detect the faces in
	 * @return Array of face bounding boxes found in the augmentation
	 */
	public FaceAnnotation[] faceDetection(BufferedImage img) {

		if (img.getType() != TYPE_INT_BGR) {
			img = this.scaleBufferedImage(img, img.getWidth(), img.getHeight());
		}

		// Stage Three
		INDArray[] stageThreeResult = this.rawFaceDetection(img);

		// Convert result into Bounding Box array
		INDArray totalBoxes = stageThreeResult[0];
		INDArray points = stageThreeResult[1];
		if (totalBoxes.size(0) > 1) {
			points = points.transpose();
		}

		return MtcnnUtil.toFaceAnnotation(totalBoxes, points);
	}

	public INDArray[] rawFaceDetection(BufferedImage img) {

		// scale pyramid
		List<Double> scales = MtcnnUtil.computeScalePyramid(img.getHeight(), img.getWidth(), this.minFaceSize, this.scaleFactor);

		// Stage One
		Object[] stageOneResult = this.preparationStage(img, scales);

		INDArray ndImage = Nd4j.create(this.imageToFloatArray(img),
				new long[] { img.getHeight(), img.getWidth(), CHANNEL_COUNT });

		// Stage Two
		INDArray totalBoxes = this.refinementStage(ndImage, (INDArray) stageOneResult[0], (MtcnnUtil.PadResult) stageOneResult[1]);

		// Stage Three
		INDArray[] stageThreeResult = this.outputStage(ndImage, totalBoxes);

		return stageThreeResult;
	}

	private Object[] preparationStage(BufferedImage image, List<Double> scales) {

		INDArray totalBoxes = Nd4j.empty();
		MtcnnUtil.PadResult padResult = null;

		for (Double scale : scales) {

			BufferedImage scaledImage = this.scaleBufferedImage(image, scale);

			INDArray ndImage = Nd4j.create(imageToFloatArray(scaledImage),
					new long[] { scaledImage.getHeight(), scaledImage.getWidth(), CHANNEL_COUNT });

			ndImage = ndImage.subi(127.5).muli(0.0078125);

			// img_x = np.expand_dims(scaled_image, 0)
			INDArray imgX = Nd4j.expandDims(ndImage, 0);

			// img_y = np.transpose(img_x, (0, 2, 1, 3))
			// NOTE: use 'dup()' to materialize the array before converting it into Tensor.
			INDArray imgY = imgX.permutei(0, 2, 1, 3).dup();

			//Map<String, Tensor<?>> output = this.proposeNet.evaluate(
			//		MtcnnUtil.input("pnet/input:0", MtcnnUtil.toTensor(imgY)),
			//		MtcnnUtil.names("pnet/conv4-2/BiasAdd:0", "pnet/prob1:0"));
			//
			//// out0 = np.transpose(out[0], (0, 2, 1, 3))
			//INDArray out0 = MtcnnUtil.toNDArray(output.get("pnet/conv4-2/BiasAdd:0"));//.permutei(0, 2, 1, 3);
			//
			//// out1 = np.transpose(out[1], (0, 2, 1, 3))
			//INDArray out1 = MtcnnUtil.toNDArray(output.get("pnet/prob1:0"));//.permutei(0, 2, 1, 3);

			Map<String, INDArray> resultMap = this.proposeNetGraphRunner.run(Collections.singletonMap("pnet/input", imgY));

			INDArray out0 = resultMap.get("pnet/conv4-2/BiasAdd");//.permutei(0, 2, 1, 3);
			INDArray out1 = resultMap.get("pnet/prob1");//.permutei(0, 2, 1, 3);

			// boxes, _ = self.__generate_bounding_box(out1[0, :, :, 1].copy(),
			//    out0[0, :, :, :].copy(), scale, self.__steps_threshold[0])
			INDArray[] bboxAndReg = MtcnnUtil.generateBoundingBox(
					out1.get(point(0), all(), all(), point(1)),
					out0.get(point(0), all(), all(), all()),
					scale, this.stepsThreshold[0]);

			INDArray boxes = bboxAndReg[0];

			if (!boxes.isEmpty()) {
				INDArray pick = MtcnnUtil.nonMaxSuppression(boxes, 0.5, Union);
				if (boxes.length() > 0 && pick.length() > 0 && !pick.isEmpty()) {
					boxes = boxes.get(new SpecifiedIndex(pick.toLongVector()), all());
					if (totalBoxes.isEmpty()) {
						totalBoxes = boxes;
					}
					else {
						totalBoxes = MtcnnUtil.append(totalBoxes, boxes, 0);
					}
				}
			}
		}

		long numBoxes = totalBoxes.shape()[0];
		if (numBoxes > 0) {
			INDArray pick = MtcnnUtil.nonMaxSuppression(totalBoxes, 0.7, Union);
			//INDArray pick = this.nonMaxSuppression(totalBoxes.dup(), 0.7, NonMaxSuppressionType.Union);
			totalBoxes = totalBoxes.get(new SpecifiedIndex(pick.toLongVector()), all());

			// regw = total_boxes[:, 2] - total_boxes[:, 0]
			// regh = total_boxes[:, 3] - total_boxes[:, 1]
			INDArray x2 = totalBoxes.get(all(), point(2));
			INDArray x1 = totalBoxes.get(all(), point(0));
			INDArray y2 = totalBoxes.get(all(), point(3));
			INDArray y1 = totalBoxes.get(all(), point(1));

			INDArray regw = x2.sub(x1);
			INDArray regh = y2.sub(y1);

			// qq1 = total_boxes[:, 0] + total_boxes[:, 5] * regw
			// qq2 = total_boxes[:, 1] + total_boxes[:, 6] * regh
			// qq3 = total_boxes[:, 2] + total_boxes[:, 7] * regw
			// qq4 = total_boxes[:, 3] + total_boxes[:, 8] * regh
			INDArray qq1 = x1.add(totalBoxes.get(all(), point(5)).mul(regw));
			INDArray qq2 = y1.add(totalBoxes.get(all(), point(6)).mul(regh));
			INDArray qq3 = x2.add(totalBoxes.get(all(), point(7)).mul(regw));
			INDArray qq4 = y2.add(totalBoxes.get(all(), point(8)).mul(regh));

			// total_boxes = np.transpose(np.vstack([qq1, qq2, qq3, qq4, total_boxes[:, 4]]))
			totalBoxes = Nd4j.hstack(qq1, qq2, qq3, qq4, totalBoxes.get(all(), point(4)));

			// total_boxes = self.__rerec(total_boxes.copy())
			totalBoxes = MtcnnUtil.rerec(totalBoxes, true);

			padResult = MtcnnUtil.pad(totalBoxes, image.getWidth(), image.getHeight());
		}

		return new Object[] { totalBoxes, padResult };
	}

	private INDArray refinementStage(INDArray image, INDArray totalBoxes, MtcnnUtil.PadResult padResult) {

		// num_boxes = total_boxes.shape[0]
		int numBoxes = (int) totalBoxes.shape()[0];
		// if num_boxes == 0:
		//   return total_boxes, stage_status
		if (numBoxes == 0) {
			return totalBoxes;
		}

		INDArray tempImg1 = computeTempImage(image, numBoxes, padResult, 24);

		if (tempImg1.isEmpty()) {
			return tempImg1;
		}

		Map<String, INDArray> resultMap = this.refineNetGraphRunner.run(Collections.singletonMap("rnet/input", tempImg1));
		INDArray out0 = resultMap.get("rnet/fc2-2/fc2-2");
		INDArray out1 = resultMap.get("rnet/prob1");

		//  score = out1[1, :]
		INDArray score = out1.get(point(1), all());

		// ipass = np.where(score > self.__steps_threshold[1])
		INDArray ipass = MtcnnUtil.getIndexWhereVector(score.transpose(), s -> s > stepsThreshold[1]);

		// total_boxes = np.hstack([total_boxes[ipass[0], 0:4].copy(), np.expand_dims(score[ipass].copy(), 1)])
		INDArray b1 = totalBoxes.get(new SpecifiedIndex(ipass.toLongVector()), interval(0, 4)).dup();
		INDArray b2 = Nd4j.expandDims(score.get(ipass).dup(), 1);
		totalBoxes = Nd4j.hstack(b1, b2);

		// mv = out0[:, ipass[0]]
		INDArray mv = out0.get(all(), new SpecifiedIndex(ipass.toLongVector()));

		// if total_boxes.shape[0] > 0:
		if (totalBoxes.shape()[0] > 0) {
			// pick = self.__nms(total_boxes, 0.7, 'Union')
			INDArray pick = MtcnnUtil.nonMaxSuppression(totalBoxes.dup(), 0.7, Union).transpose();

			// total_boxes = total_boxes[pick, :]
			totalBoxes = totalBoxes.get(new SpecifiedIndex(pick.toLongVector()), all());

			// total_boxes = self.__bbreg(total_boxes.copy(), np.transpose(mv[:, pick]))
			totalBoxes = MtcnnUtil.bbreg(totalBoxes.dup(), mv.get(all(), new SpecifiedIndex(pick.toLongVector())).transpose());

			// total_boxes = self.__rerec(total_boxes.copy())
			totalBoxes = MtcnnUtil.rerec(totalBoxes.dup(), false);
		}

		return totalBoxes;
	}

	private INDArray[] outputStage(INDArray image, INDArray totalBoxes) {
		// num_boxes = total_boxes.shape[0]
		int numBoxes = (int) totalBoxes.shape()[0];
		// if num_boxes == 0:
		//   return total_boxes, stage_status
		if (numBoxes == 0) {
			return new INDArray[] { totalBoxes, Nd4j.empty() };
		}

		// total_boxes = np.fix(total_boxes).astype(np.int32)
		totalBoxes = Transforms.floor(totalBoxes);

		// status = StageStatus(self.__pad(total_boxes.copy(), stage_status.width, stage_status.height),
		//                             width=stage_status.width, height=stage_status.height)
		MtcnnUtil.PadResult padResult = MtcnnUtil.pad(totalBoxes, (int) image.shape()[0], (int) image.shape()[1]);

		INDArray tempImg1 = computeTempImage(image, numBoxes, padResult, 48);

		if (tempImg1.isEmpty()) {
			return new INDArray[] { Nd4j.empty(), Nd4j.empty() };
		}

		Map<String, INDArray> resultMap = this.outputNetGraphRunner.run(Collections.singletonMap("onet/input", tempImg1));
		INDArray out0 = resultMap.get("onet/fc2-2/fc2-2");
		INDArray out1 = resultMap.get("onet/fc2-3/fc2-3");
		INDArray out2 = resultMap.get("onet/prob1");

		// score = out2[1, :]
		INDArray score = out2.get(point(1), all());

		// points = out1
		INDArray points = out1;

		// ipass = np.where(score > self.__steps_threshold[2])
		INDArray ipass = MtcnnUtil.getIndexWhereVector(score.transpose(), s -> s > stepsThreshold[2]);

		// points = points[:, ipass[0]]
		points = points.get(all(), new SpecifiedIndex(ipass.toLongVector()));

		// total_boxes = np.hstack([total_boxes[ipass[0], 0:4].copy(), np.expand_dims(score[ipass].copy(), 1)])
		INDArray b1 = totalBoxes.get(new SpecifiedIndex(ipass.toLongVector()), interval(0, 4)).dup();
		INDArray b2 = Nd4j.expandDims(score.get(ipass).dup(), 1);
		totalBoxes = Nd4j.hstack(b1, b2);

		// mv = out0[:, ipass[0]]
		INDArray mv = out0.get(all(), new SpecifiedIndex(ipass.toLongVector()));

		//  w = total_boxes[:, 2] - total_boxes[:, 0] + 1
		//  h = total_boxes[:, 3] - total_boxes[:, 1] + 1
		INDArray w = totalBoxes.get(all(), point(2)).dup().subi(totalBoxes.get(all(), point(0))).addi(1);
		INDArray h = totalBoxes.get(all(), point(3)).dup().subi(totalBoxes.get(all(), point(1))).addi(1);

		// points[0:5, :] = np.tile(w, (5, 1)) * points[0:5, :] + np.tile(total_boxes[:, 0], (5, 1)) - 1
		// points[5:10, :] = np.tile(h, (5, 1)) * points[5:10, :] + np.tile(total_boxes[:, 1], (5, 1)) - 1
		points.put(new INDArrayIndex[] { interval(0, 5), all() },
				Nd4j.repeat(w, 5)
						.muli(points.get(interval(0, 5), all()))
						.addi(Nd4j.repeat(totalBoxes.get(all(), point(0)), 5))
						.subi(1));

		points.put(new INDArrayIndex[] { interval(5, 10), all() },
				Nd4j.repeat(h, 5)
						.muli(points.get(interval(5, 10), all()))
						.addi(Nd4j.repeat(totalBoxes.get(all(), point(1)), 5))
						.subi(1));

		if (totalBoxes.shape()[0] > 0) {

			// total_boxes = self.__bbreg(total_boxes.copy(), np.transpose(mv))
			totalBoxes = MtcnnUtil.bbreg(totalBoxes.dup(), mv.transpose());

			// pick = self.__nms(total_boxes.copy(), 0.7, 'Min')
			INDArray pick = MtcnnUtil.nonMaxSuppression(totalBoxes.dup(), 0.7, Min).transpose();

			//  total_boxes = total_boxes[pick, :]
			totalBoxes = totalBoxes.get(new SpecifiedIndex(pick.toLongVector()), all());

			// points = points[:, pick]
			points = points.get(all(), new SpecifiedIndex(pick.toLongVector()));
		}

		return new INDArray[] { totalBoxes, points };
	}

	private INDArray computeTempImage(INDArray image, int numBoxes, MtcnnUtil.PadResult padResult, int size) {

		// tempimg = np.zeros((48, 48, 3, num_boxes))
		INDArray tempImg = Nd4j.zeros(new int[] { size, size, CHANNEL_COUNT, numBoxes }, C_ORDERING);

		for (int k = 0; k < numBoxes; k++) {
			// tmp = np.zeros((int(status.tmph[k]), int(status.tmpw[k]), 3))
			INDArray tmp = Nd4j.zeros(new int[] { padResult.getTmph().getInt(k), padResult.getTmpw().getInt(k), CHANNEL_COUNT }, C_ORDERING);

			// tmp[status.dy[k] - 1:status.edy[k], status.dx[k] - 1:status.edx[k], :] = \
			//   img[status.y[k] - 1:status.ey[k], status.x[k] - 1:status.ex[k], :]
			tmp.put(new INDArrayIndex[] {
							interval(padResult.getDy().getInt(k) - 1, padResult.getEdy().getInt(k)),
							interval(padResult.getDx().getInt(k) - 1, padResult.getEdx().getInt(k)),
							all() },
					image.get(
							interval(padResult.getY().getInt(k) - 1, padResult.getEy().getInt(k)),
							interval(padResult.getX().getInt(k) - 1, padResult.getEx().getInt(k)),
							all()));

			// if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
			//   tempimg[:, :, :, k] = cv2.resize(tmp, (48, 48), interpolation=cv2.INTER_AREA)
			//                                   OR
			// tmp[stage_status.dy[k] - 1:stage_status.edy[k], stage_status.dx[k] - 1:stage_status.edx[k], :] = \
			//   img[stage_status.y[k] - 1:stage_status.ey[k], stage_status.x[k] - 1:stage_status.ex[k], :]
			if ((tmp.shape()[0] > 0 && tmp.shape()[1] > 0) || (tmp.shape()[0] == 0 && tmp.shape()[1] == 0)) {
				INDArray resizedImage = this.scaleNDArrayImage(tmp, size, size);
				tempImg.put(new INDArrayIndex[] { all(), all(), all(), point(k) }, resizedImage);
			}
			else {
				return Nd4j.empty();
			}
		}

		// tempimg = (tempimg - 127.5) * 0.0078125
		tempImg = tempImg.subi(127.5).muli(0.0078125);

		// tempimg1 = np.transpose(tempimg, (3, 1, 0, 2))
		// Flips the X and Y axes (e.g. from 0,1,2,3 to 3,1,0,2
		// NOTE: use 'dup()' to materialize the array before converting it into Tensor.
		INDArray tempImg1 = tempImg.permutei(3, 1, 0, 2).dup();

		return tempImg1;
	}

	//private byte[] toByteArray(BufferedImage bufferedImage) {
	//	return ((DataBufferByte) bufferedImage.getRaster().getDataBuffer()).getData();
	//}

	//private float[] imageToFloatArray(BufferedImage bufferedImage) {
	//	return imageToFloatArray(toByteArray(bufferedImage));
	//}

	//private float[] imageToFloatArray(byte[] imageBytes) {
	//	float[] fa = new float[imageBytes.length];
	//	for (int i = 0; i < imageBytes.length; i++) {
	//		fa[i] = imageBytes[i] & 0xFF;
	//	}
	//	return fa;
	//}

	private float[] imageToFloatArray(BufferedImage bi) {
		int[] ia = ((DataBufferInt) bi.getRaster().getDataBuffer()).getData();
		float[] fa = new float[ia.length * CHANNEL_COUNT];
		for (int i = 0; i < ia.length; i++) {
			final int val = ia[i];
			fa[i * CHANNEL_COUNT + COLOR_CHANNELS[0]] = ((val >> 16) & 0xFF); //R
			fa[i * CHANNEL_COUNT + COLOR_CHANNELS[1]] = ((val >> 8) & 0xFF);  //G
			fa[i * CHANNEL_COUNT + COLOR_CHANNELS[2]] = (val & 0xFF);         //B
		}
		return fa;
	}

	private BufferedImage scaleBufferedImage(BufferedImage originalImage, double scale) {

		int newWidth = (int) Math.ceil(originalImage.getWidth() * scale);
		int newHeight = (int) Math.ceil(originalImage.getHeight() * scale);

		return this.scaleBufferedImage(originalImage, newWidth, newHeight);
	}

	public BufferedImage scaleBufferedImage(BufferedImage originalImage, int newWidth, int newHeight) {

		BufferedImage resizedImage = new BufferedImage(newWidth, newHeight, TYPE_INT_BGR);
		Graphics2D g2d = resizedImage.createGraphics();
		g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
		g2d.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_SPEED);

		g2d.drawImage(originalImage, 0, 0, newWidth, newHeight, null);
		g2d.dispose();

		return resizedImage;
	}

	/**
	 * Scales INDArray augmentation.
	 *  This implementation requires conversion to and from BufferedImage
	 *
	 * @param ndImage input INDArray augmentation
	 * @param width target augmentation width
	 * @param height target augmentation height
	 * @return Returns the resized INDArray
	 */
	private INDArray scaleNDArrayImage(INDArray ndImage, int width, int height) {

		// 1. Convert INDArray augmentation into  BufferedImage. Sum up the R, G, B byte columns into a single INT column.
		// E.g. (R + G * 8 + B * 16) -> int matrix -> int[][] -> flatten -> int[]
		int[] imagePixels = ArrayUtil.flatten(
				ndImage.get(all(), all(), point(0)) // << 0
						.addi(ndImage.get(all(), all(), point(1)).muli(256)) // << 8
						.addi(ndImage.get(all(), all(), point(2)).muli(65536)) // << 16
						.toIntMatrix());

		int[] shape = ArrayUtil.toInts(ndImage.shape());
		BufferedImage bufferedImage = new BufferedImage(shape[1], shape[0], BufferedImage.TYPE_INT_RGB);
		bufferedImage.setRGB(0, 0, shape[1], shape[0], imagePixels, 0, shape[1]);

		// 2. Scale the buffered augmentation
		BufferedImage scaledBufferedImage = this.scaleBufferedImage(bufferedImage, width, height);

		// testWriteImage(scaledBufferedImage, "scaleNDArrayImage");

		// 3. Convert the BufferedImage back to float INDArray
		INDArray result = Nd4j.create(imageToFloatArray(scaledBufferedImage), new long[] { height, width, CHANNEL_COUNT });


		return result;
	}

	//private BufferedImage scaleBufferedImage(BufferedImage originalImage, int newWidth, int newHeight) {
	//
	//	Image tmpImage = originalImage.getScaledInstance(newWidth, newHeight, Image.SCALE_SMOOTH);
	//	//Image tmpImage = originalImage.getScaledInstance(newWidth, newHeight, Image.SCALE_FAST);
	//	BufferedImage resizedImage = new BufferedImage(newWidth, newHeight, TYPE_3BYTE_BGR);
	//
	//	Graphics2D g2d = resizedImage.createGraphics();
	//	g2d.drawImage(tmpImage, 0, 0, null);
	//	g2d.dispose();
	//
	//	return resizedImage;
	//}

	//private BufferedImage scaleBufferedImage(BufferedImage originalImage, int newWidth, int newHeight) {
	//
	//	// Ensure that the type is 3-byte BGR
	//	BufferedImage resizedImage = new BufferedImage(newWidth, newHeight, TYPE_3BYTE_BGR);
	//
	//	Graphics2D g2d = resizedImage.createGraphics();
	//	g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
	//	g2d.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_SPEED);
	//
	//	g2d.drawImage(originalImage, 0, 0, newWidth, newHeight, null);
	//	g2d.dispose();
	//
	//	return resizedImage;
	//}

	private static int cnt = 0;

	private void testWriteImage(BufferedImage image, String prefix) {
		try {
			ImageIO.write(image, "png",
					new File("spring-cloud-starter-stream-processor-face-detection-mtcnn/target/test" + prefix + cnt + ".png"));
			cnt++;
			System.out.println(".");
		}
		catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static void main(String[] args) throws IOException {
		MtcnnServiceJava2D mtcnnService = new MtcnnServiceJava2D();
		String imageUri = "file:spring-cloud-starter-stream-processor-face-detection-mtcnn/src/test/resources/ivan.jpg";
		String imageUri2 = "file:spring-cloud-starter-stream-processor-face-detection-mtcnn/src/test/resources/VikiMaxiAdi.jpg";
		String imageUri3 = "file:spring-cloud-starter-stream-processor-face-detection-mtcnn/src/test/resources/VikiMaxiAdi.jpg";
		BufferedImage img = ImageIO.read(
				new DefaultResourceLoader().getResource(imageUri).getInputStream());
		img = mtcnnService.scaleBufferedImage(img, img.getWidth(), img.getHeight());


//		for (int i = 0; i < 1000; i++) {
		FaceAnnotation[] boundingBoxes = mtcnnService.faceDetection(img);
		System.out.println(boundingBoxes.length);
		System.out.println("Result: " + new ObjectMapper().writeValueAsString(boundingBoxes));
//		}
	}
}
