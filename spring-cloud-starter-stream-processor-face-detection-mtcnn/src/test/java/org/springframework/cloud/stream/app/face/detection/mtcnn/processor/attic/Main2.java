package org.springframework.cloud.stream.app.face.detection.mtcnn.processor.attic;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.io.IOException;

import javax.imageio.ImageIO;

import org.springframework.cloud.stream.app.tensorflow.util.GraphicsUtils;
import org.springframework.core.io.DefaultResourceLoader;
import org.springframework.util.Assert;

/**
 * @author Christian Tzolov
 */
public class Main2 {


	public static void main(String[] args) throws IOException {
		boza2();
	}

	public static void boza2() {
		final int[] original = new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
		int w = 4;
		int h = 3;

		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				System.out.print(original[x + y * w] + " ");
			}
			System.out.print("\n");
		}

		int[] transposed = transpose(original, w, h);

		int w2 = h;
		int h2 = w;
		for (int y = 0; y < h2; y++) {
			for (int x = 0; x < w2; x++) {
				System.out.print(transposed[x + y * w2] + " ");
			}
			System.out.print("\n");
		}

	}

	private static int[] transpose(int[] input, int w, int h) {

		Assert.isTrue(input.length == h * w, "Mismatch");
		int[] output = new int[h * w];
		for (int y = h - 1; y >= 0; y--) {
			for (int x = 0; x < w; x++) {
				output[y + x * h] = input[x + y * w];
			}
		}
		return output;
	}

	public static void boza3() throws IOException {
		BufferedImage img = ImageIO.read(
				new DefaultResourceLoader().getResource("classpath:/ivan.jpg").getInputStream());

		System.out.println(img.getWidth() * img.getHeight());

		BufferedImage imgToRecognition = GraphicsUtils.toBufferedImage(img);

		int[] bb2 = ((DataBufferInt) imgToRecognition.getRaster().getDataBuffer()).getData();


		System.out.println(bb2.length);

		for (int x = 0; x < img.getWidth(); x++) {
			for (int y = 0; y < img.getHeight(); y++) {
				System.out.print(imgToRecognition.getRGB(x, y) + " (" + bb2[x + y * img.getHeight()] + ") ");
				Assert.isTrue(imgToRecognition.getRGB(x, y) == bb2[x + y * img.getHeight()],
						"Not equal for " + x + ":" + y);
			}
			System.out.println();
		}
		System.out.println();

	}

	public static void boza() {
		final int[][] original = new int[][] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 }, { 9, 10, 11, 12 } };
		for (int i = 0; i < original.length; i++) {
			for (int j = 0; j < original[i].length; j++) {
				System.out.print(original[i][j] + " ");
			}
			System.out.print("\n");
		}

		System.out.println();
		int[][] transposed = rotate(original);

		for (int i = 0; i < transposed.length; i++) {
			for (int j = 0; j < transposed[0].length; j++) {
				System.out.print(transposed[i][j] + " ");
			}
			System.out.print("\n");
		}
	}

	/**
	 * rotate clockwise in 90 degree
	 * @param input The 2D matrix to be rotated
	 * @return The input matrix rotated clockwise in 90 degrees
	 */
	private static int[][] rotate(int[][] input) {

		int w = input.length;
		int h = input[0].length;

		int[][] output = new int[h][w];
		for (int y = 0; y < h; y++) {
			for (int x = w - 1; x >= 0; x--) {
				output[y][x] = input[x][y];
			}
		}
		return output;
	}
}
