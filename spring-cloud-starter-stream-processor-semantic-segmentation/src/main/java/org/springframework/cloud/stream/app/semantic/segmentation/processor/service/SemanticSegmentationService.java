package org.springframework.cloud.stream.app.semantic.segmentation.processor.service;

import java.awt.image.BufferedImage;

import org.tensorflow.Tensor;
import org.tensorflow.types.UInt8;

/**
 * @author Christian Tzolov
 */
public interface SemanticSegmentationService<T> {

	String INPUT_TENSOR_NAME = "ImageTensor:0";
	String OUTPUT_TENSOR_NAME = "SemanticPredictions:0";


	T scaledImage(String imagePath);

	T scaledImage(byte[] image);

	T scaledImage(BufferedImage image);

	T blendMask(T mask, T background);

	Tensor<UInt8> createInputTensor(T resizedImage);

	T createMaskImage(int[][] maskPixels, int width, int height, double transparency);

	default int[][] toIntArray(long[][] longArray) {
		int[][] intArray = new int[longArray.length][longArray[0].length];
		for (int i = 0; i < longArray.length; i++) {
			for (int j = 0; j < longArray[0].length; j++) {
				intArray[i][j] = (int) longArray[i][j];
			}
		}
		return intArray;
	}

}
