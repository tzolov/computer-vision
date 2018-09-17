package org.springframework.cloud.stream.app.face.recognition.processor;

import java.util.Arrays;

import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.core.Mul;

/**
 * @author Christian Tzolov
 */
public class Main2 {
	public static void main(String[] args) {

		try (Graph g = new Graph()) {

			Ops ops = Ops.create(g);

			//Constant<Integer> aaa = ops.constant(new int[][] { { 1, 2, 3, 4 }, { 1, 2, 3, 4 } }, Integer.class);
			//
			//Operand mask = ops.greater(aaa, ops.constant(3, Integer.class));
			//Operand maskIdx = ops.where(mask);
			//Operand matchValues = ops.gatherNd(aaa, maskIdx);



			Constant<Integer> aa = ops.constant(new int[][] { { 1, 1 }, { 2, 2 }, { 3, 3 }, { 4, 4 } }, Integer.class);
			Constant<Integer> bb = ops.constant(new int[][] { { 1, 1 }, { 2, 2 }, { 3, 3 }, { 4, 4 } }, Integer.class);
			Mul<Integer> cc = ops.mul(aa, bb);

			Constant<Integer> a = ops.constant(3, Integer.class);
			Constant<Integer> b = ops.constant(5, Integer.class);
			Operand<Integer> c = ops.mul(a, b);


			try (Session s = new Session(g)) {
				// Generally, there may be multiple output tensors, all of them must be closed to prevent resource leaks.

				Tensor<Integer> d = s.runner().fetch(c.asOutput().op().name()).run().get(0).expect(Integer.class);
				//Tensor<Long> d = s.runner().fetch(maskIdx.asOutput().op().name()).run().get(0).expect(Long.class);
				//Tensor<Integer> d = s.runner().fetch(indxes.asOutput().op().name()).run().get(0).expect(Integer.class);
				//Tensor<Boolean> d = s.runner().fetch(mask.asOutput().op().name()).run().get(0).expect(Boolean.class);
				//int i = d.intValue();
				System.out.println(Arrays.toString(d.shape()));
				System.out.println(d.intValue());
				//long[][] result = new long[4][2];

				//int[][] result = new int[4][2];
				//boolean[][] result = new boolean[4][2];
				//d.copyTo(result);

//				System.out.println(Arrays.toString(to.int1(d)));
			}
		}
	}

	public static class to {

		private static int dim(Tensor<?> tensor, int dimension) {
			return (int) tensor.shape()[dimension];
		}

		// int
		public static int[] int1(Tensor<Integer> tensor) {
			int[] array = new int[dim(tensor,0)];
			return tensor.copyTo(array);
		}

		public static int[][] int2(Tensor<Integer> tensor) {
			int[][] array = new int[dim(tensor,0)][dim(tensor,1)];
			return tensor.copyTo(array);
		}

		public static int[][][] int3(Tensor<Integer> tensor) {
			int[][][] array = new int[dim(tensor,0)][dim(tensor,1)][dim(tensor,2)];
			return tensor.copyTo(array);
		}

		public static int[][][][] int4(Tensor<Integer> tensor) {
			int[][][][] array = new int[dim(tensor,0)][dim(tensor,1)][dim(tensor,2)][dim(tensor,3)];
			return tensor.copyTo(array);
		}

		public static int[][][][][] int5(Tensor<Integer> tensor) {
			int[][][][][] array = new int[dim(tensor,0)][dim(tensor,1)][dim(tensor,2)][dim(tensor,3)][dim(tensor,4)];
			return tensor.copyTo(array);
		}

		// long
		public static long[] long1(Tensor<Long> tensor) {
			long[] array = new long[dim(tensor,0)];
			return tensor.copyTo(array);
		}

		public static long[][] long2(Tensor<Long> tensor) {
			long[][] array = new long[dim(tensor,0)][dim(tensor,1)];
			return tensor.copyTo(array);
		}

		public static long[][][] long3(Tensor<Long> tensor) {
			long[][][] array = new long[dim(tensor,0)][dim(tensor,1)][dim(tensor,2)];
			return tensor.copyTo(array);
		}

		public static long[][][][] long4(Tensor<Long> tensor) {
			long[][][][] array = new long[dim(tensor,0)][dim(tensor,1)][dim(tensor,2)][dim(tensor,3)];
			return tensor.copyTo(array);
		}

		public static long[][][][][] long5(Tensor<Long> tensor) {
			long[][][][][] array = new long[dim(tensor,0)][dim(tensor,1)][dim(tensor,2)][dim(tensor,3)][dim(tensor,4)];
			return tensor.copyTo(array);
		}

		// float
		public static float[] float1(Tensor<Float> tensor) {
			float[] array = new float[dim(tensor,0)];
			return tensor.copyTo(array);
		}

		public static float[][] float2(Tensor<Float> tensor) {
			float[][] array = new float[dim(tensor,0)][dim(tensor,1)];
			return tensor.copyTo(array);
		}

		public static float[][][] float3(Tensor<Float> tensor) {
			float[][][] array = new float[dim(tensor,0)][dim(tensor,1)][dim(tensor,2)];
			return tensor.copyTo(array);
		}

		public static float[][][][] float4(Tensor<Float> tensor) {
			float[][][][] array = new float[dim(tensor,0)][dim(tensor,1)][dim(tensor,2)][dim(tensor,3)];
			return tensor.copyTo(array);
		}

		public static float[][][][][] float5(Tensor<Float> tensor) {
			float[][][][][] array = new float[dim(tensor,0)][dim(tensor,1)][dim(tensor,2)][dim(tensor,3)][dim(tensor,4)];
			return tensor.copyTo(array);
		}

		// double
		public static double[] double1(Tensor<Double> tensor) {
			double[] array = new double[dim(tensor,0)];
			return tensor.copyTo(array);
		}

		public static double[][] double2(Tensor<Double> tensor) {
			double[][] array = new double[dim(tensor,0)][dim(tensor,1)];
			return tensor.copyTo(array);
		}

		public static double[][][] double3(Tensor<Double> tensor) {
			double[][][] array = new double[dim(tensor,0)][dim(tensor,1)][dim(tensor,2)];
			return tensor.copyTo(array);
		}

		public static double[][][][] double4(Tensor<Double> tensor) {
			double[][][][] array = new double[dim(tensor,0)][dim(tensor,1)][dim(tensor,2)][dim(tensor,3)];
			return tensor.copyTo(array);
		}

		public static double[][][][][] double5(Tensor<Double> tensor) {
			double[][][][][] array = new double[dim(tensor,0)][dim(tensor,1)][dim(tensor,2)][dim(tensor,3)][dim(tensor,4)];
			return tensor.copyTo(array);
		}


		// boolean
		public static boolean[] boolean1(Tensor<Boolean> tensor) {
			boolean[] array = new boolean[dim(tensor,0)];
			return tensor.copyTo(array);
		}

		public static boolean[][] boolean2(Tensor<Boolean> tensor) {
			boolean[][] array = new boolean[dim(tensor,0)][dim(tensor,1)];
			return tensor.copyTo(array);
		}

		public static boolean[][][] boolean3(Tensor<Boolean> tensor) {
			boolean[][][] array = new boolean[dim(tensor,0)][dim(tensor,1)][dim(tensor,2)];
			return tensor.copyTo(array);
		}

		public static boolean[][][][] boolean4(Tensor<Boolean> tensor) {
			boolean[][][][] array = new boolean[dim(tensor,0)][dim(tensor,1)][dim(tensor,2)][dim(tensor,3)];
			return tensor.copyTo(array);
		}

		public static boolean[][][][][] boolean5(Tensor<Boolean> tensor) {
			boolean[][][][][] array = new boolean[dim(tensor,0)][dim(tensor,1)][dim(tensor,2)][dim(tensor,3)][dim(tensor,4)];
			return tensor.copyTo(array);
		}

	}

}
