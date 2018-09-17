package org.springframework.cloud.stream.app.face.detection.mtcnn.processor.attic;

import java.util.Arrays;

import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Conditions;

/**
 * @author Christian Tzolov
 */
public class IndexWhereTest {

	//@Test
	//public void whereWithEmptyMask() {
	//	INDArray inArray = Nd4j.zeros(2, 3);
	//	inArray.putScalar(0, 0, 10.0f);
	//	inArray.putScalar(1, 2, 10.0f);
	//
	//	INDArray mask1 = inArray.match(1, Conditions.greaterThanOrEqual(1));
	//
	//	Assert.assertTrue(mask1.maxNumber().intValue() == 1); // ! Not Empty Match
	//
	//	INDArray[] matchIndexes = Nd4j.where(mask1, null, null);
	//
	//	Assert.assertArrayEquals(new int[] {0, 1}, matchIndexes[0].toIntVector());
	//	Assert.assertArrayEquals(new int[] {0, 2}, matchIndexes[1].toIntVector());
	//
	//	INDArray mask2 = inArray.match(1, Conditions.greaterThanOrEqual(11));
	//
	//	Assert.assertTrue(mask2.maxNumber().intValue() == 0);
	//
	//	INDArray[] matchIndexes2 = Nd4j.where(mask2, null, null);
	//}
	//
	//public static void main(String[] args) {
	//	INDArray inArray = Nd4j.zeros(2, 3);
	//	inArray.putScalar(0, 0, 666.0f);
	//	inArray.putScalar(1, 2, 666.0f);
	//
	//	System.out.println("Input: \n" + inArray);
	//
	//	INDArray mask = inArray.match(1, Conditions.greaterThanOrEqual(1));
	//
	//	System.out.println("Mask: \n" + mask);
	//
	//	INDArray[] matchIndexes = Nd4j.where(mask, null, null);
	//
	//	System.out.println("Indexes length: " + matchIndexes.length);
	//	System.out.println("Indexes shape: " + Arrays.toString(matchIndexes[0].shape()));
	//	System.out.println("Indexes [0]: " + matchIndexes[0]);
	//	System.out.println("Indexes [1]: " + matchIndexes[1]);
	//}
}
