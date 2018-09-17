package org.springframework.cloud.stream.app.face.detection.mtcnn.processor.attic;

import java.io.IOException;
import java.nio.FloatBuffer;

import org.bytedeco.javacpp.BoolPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgcodecs;
import org.bytedeco.javacpp.tensorflow;
import org.datavec.image.loader.Java2DNativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;

import org.springframework.core.io.DefaultResourceLoader;

import static org.bytedeco.javacpp.tensorflow.InitMain;

/**
 *
 * use a pre-trained model from: https://github.com/davidsandberg/facenet#pre-trained-models
 * As reference how to inference check: https://github.com/davidsandberg/facenet/blob/master/src/compare.py
 *
 * @author Christian Tzolov
 */
public class FaceNetTest2 {


	public static void main(String[] args) throws IOException {

		// Load all javacpp-preset classes and native libraries
		Loader.load(org.bytedeco.javacpp.tensorflow.class);

		// Platform-specific initialization routine
		InitMain("trainer", (int[])null, null);

		final tensorflow.Session session = new tensorflow.Session(new tensorflow.SessionOptions());

		tensorflow.GraphDef def = new tensorflow.GraphDef();
		tensorflow.ReadBinaryProto(tensorflow.Env.Default(), "/Users/ctzolov/Downloads/20180408-102900/20180408-102900.pb", def);
		tensorflow.Status s = session.Create(def);
		if (!s.ok()) {
			throw new RuntimeException(s.error_message().getString());
		}

		try (PointerScope scope = new PointerScope()) {

			//opencv_core.Mat mat = opencv_imgcodecs.imread(
			//		"/Users/ctzolov/Dev/projects/scdf/computer-vision/spring-cloud-starter-stream-processor-face-detection-mtcnn/target/cropped0.png");
			//
			INDArray image = new Java2DNativeImageLoader().asMatrix(
					new DefaultResourceLoader().getResource("file:/Users/ctzolov/Dev/projects/scdf/computer-vision/spring-cloud-starter-stream-processor-face-detection-mtcnn/target/cropped0.png").getInputStream())
					.permutei(0, 2, 3, 1).dup();

			opencv_core.Mat mat = new Java2DNativeImageLoader().asMat(image);
			tensorflow.Tensor imgTensor = new tensorflow.Tensor(tensorflow.DT_FLOAT,
					new tensorflow.TensorShape(1, mat.rows(), mat.cols(), mat.channels()),
					mat.arrayData().capacity(mat.arraySize()));

			BoolPointer bp = new BoolPointer(1);
			bp.put(false);
			tensorflow.Tensor boolTensor = new tensorflow.Tensor(tensorflow.DT_BOOL, new tensorflow.TensorShape(1), bp);

			tensorflow.TensorVector outputs = new tensorflow.TensorVector();

			s = session.Run(
					new tensorflow.StringTensorPairVector(new String[] { "input", "phase_train" }, new tensorflow.Tensor[] { imgTensor, boolTensor }),
					new tensorflow.StringVector(new String[] { "embeddings" }), new tensorflow.StringVector(), outputs);

			if (!s.ok()) {
				throw new RuntimeException(s.error_message().getString());
			}

			tensorflow.Tensor to = outputs.get(0);
			FloatBuffer fb = to.createBuffer();
			float f1 = fb.get(0);
			float[] aa = fb.array();
			System.out.println(to);
		}
	}
}
