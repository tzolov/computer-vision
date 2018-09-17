package org.springframework.cloud.stream.app.face.detection.mtcnn.processor.attic;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.io.IOUtils;
import org.bytedeco.javacpp.tensorflow;
import org.datavec.image.loader.Java2DNativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.tensorflow.conversion.graphrunner.GraphRunner;
import org.tensorflow.framework.ConfigProto;

import org.springframework.core.io.DefaultResourceLoader;

/**
 *
 * use a pre-trained model from: https://github.com/davidsandberg/facenet#pre-trained-models
 * As reference how to inference check: https://github.com/davidsandberg/facenet/blob/master/src/compare.py
 *
 * @author Christian Tzolov
 */
public class FaceNetTest {


	public static void main(String[] args) throws IOException {

		final String imageUri = "file:/Users/ctzolov/Dev/projects/scdf/computer-vision/spring-cloud-starter-stream-processor-face-detection-mtcnn/target/cropped0.png";

		final String tfModelUri = "file:/Users/ctzolov/Downloads/20180408-102900/20180408-102900.pb";

		// Image is loaded in [N, W, H, C] shape
		INDArray image = new Java2DNativeImageLoader().asMatrix(new DefaultResourceLoader().getResource(imageUri).getInputStream())
				.permutei(0, 2, 3, 1).dup();

		// load the CASIA-WebFace model (https://github.com/davidsandberg/facenet#pre-trained-models)
		byte[] frozenModelBytes = IOUtils.toByteArray(new DefaultResourceLoader().getResource(tfModelUri).getInputStream());

		GraphRunner faceNetGraph = new GraphRunner(frozenModelBytes, Arrays.asList("input", "phase_train"), ConfigProto.getDefaultInstance());

		Map<String, INDArray> inputMap = new HashMap();
		inputMap.put("input", image);
		//inputMap.put("phase_train", Nd4j.create(new float[] { 1.0f })); // MUST BE A Boolean Tensor (e.g. Tensor.create(false))
		inputMap.put("phase_train", Nd4j.scalar(0.0f)); // Doesn't work either

		Map<String, INDArray> resultMap = faceNetGraph.run(inputMap);

		System.out.println(resultMap);
	}
}
