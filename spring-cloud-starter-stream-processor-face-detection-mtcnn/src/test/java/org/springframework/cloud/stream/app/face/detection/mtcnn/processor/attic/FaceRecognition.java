package org.springframework.cloud.stream.app.face.detection.mtcnn.processor.attic;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.io.IOUtils;
import org.bytedeco.javacpp.BoolPointer;
import org.bytedeco.javacpp.tensorflow;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.tensorflow.conversion.TensorflowConversion;;
import org.tensorflow.Tensor;
import org.tensorflow.framework.ConfigProto;

import org.springframework.core.io.DefaultResourceLoader;

import org.nd4j.tensorflow.conversion.graphrunner.GraphRunner;
/**
 * @author Christian Tzolov
 */
public class FaceRecognition {

	private final GrapphRunner2 faceNet;

	public FaceRecognition() {
		faceNet = new GrapphRunner2(getFrozenModel("file:/Users/ctzolov/Downloads/20180408-102900/20180408-102900.pb"),
				Arrays.asList("input", "phase_train"), ConfigProto.getDefaultInstance());
	}

	public void run(INDArray image) {
		Map<String, tensorflow.TF_Tensor> in = new HashMap();
		in.put("input", TensorflowConversion.getInstance().tensorFromNDArray(image));

		BoolPointer bp = new BoolPointer(2);
		bp.put(false);
		tensorflow.TF_Tensor tftf = tensorflow.TF_Tensor.newTensor(10, new long[] { 1L }, bp);

		in.put("phase_train", tftf);
		//Map<String, INDArray> in = new HashMap();
		//in.put("input", augmentation);
		//in.put("phase_train", Nd4j.scalar(0));

		Tensor<?> tf = Tensor.create(false);

		Map<String, INDArray> resultMap = this.faceNet.run(in);

		//INDArray out0 = resultMap.get("pnet/conv4-2/BiasAdd");//.permutei(0, 2, 1, 3);
		//INDArray out1 = resultMap.get("pnet/prob1");//.permutei(0, 2, 1, 3);

	}

	public void run2(INDArray image) {
		GraphRunner faceNet2 = new GraphRunner(getFrozenModel("file:/Users/ctzolov/Downloads/20180408-102900/20180408-102900.pb"),
				Arrays.asList("input", "phase_train"), ConfigProto.getDefaultInstance());

		Map<String, INDArray> in = new HashMap();
		in.put("input", image);
		in.put("phase_train", Nd4j.create(new float[] {1.0f}));

		Map<String, INDArray> resultMap = faceNet2.run(in);

		System.out.println(resultMap);
	}


	private byte[] getFrozenModel(String modelUri) {
		try {
			return IOUtils.toByteArray(new DefaultResourceLoader().getResource(modelUri).getInputStream());
		}
		catch (IOException e) {
			throw new IllegalStateException("Failed to load [" + modelUri + "] resource");
		}
	}

}
