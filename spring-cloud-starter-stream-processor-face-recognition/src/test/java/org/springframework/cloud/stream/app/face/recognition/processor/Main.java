package org.springframework.cloud.stream.app.face.recognition.processor;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.Arrays;

import javax.imageio.ImageIO;

import org.springframework.cloud.stream.app.tensorflow.processor.TensorFlowService;
import org.springframework.core.io.DefaultResourceLoader;
import org.springframework.util.Assert;

/**
 * @author Christian Tzolov
 */
public class Main {

	public static void main(String[] args) throws IOException {

		final String imageUri = "file:/Users/ctzolov/Dev/projects/mtcnn/target/cropped0.png";

		final String modelUri = "file:/Users/ctzolov/Downloads/20180408-102900/20180408-102900.pb";

		TensorFlowService tfService = new TensorFlowService(new DefaultResourceLoader().getResource(modelUri));

		FaceRecognitionService faceRecognitionService = new FaceRecognitionService(tfService);

		BufferedImage bi = ImageIO.read(new DefaultResourceLoader().getResource(imageUri).getInputStream());


		float[] embeddings = faceRecognitionService.encode(bi);

		System.out.println("Test (" + embeddings.length + ") =" + Arrays.toString(embeddings));

	}

	public static double distance(float[] emb1, float[] emb2) {
		Assert.isTrue(emb1.length == emb2.length, "Compared arrays must be of same size, but " +
				"emb1: " + emb1.length + " and emb2: " + emb2.length);
		double sum = 0;
		for (int i = 0; i < emb1.length; i++) {
			sum = (emb1[i] - emb2[i]) * (emb1[i] - emb2[i]);
		}
		return Math.sqrt(sum);
	}
}
