/*
 * Copyright 2018 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.springframework.cloud.stream.app.face.detection.mtcnn.processor;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.validation.annotation.Validated;

/**
 * @author Christian Tzolov
 */
@ConfigurationProperties("face.detection.mtcnn")
@Validated
public class FaceDetectionMtcnnProcessorProperties {

	public enum OutputMode {
		/**
		 * Outbound payload contains the face annotations encoded in json format.
		 */
		annotation,
		/**
		 * Outbound message wraps the face annotations as json header, while the payload contains the inbound
		 * image possibly augmented with the face annotations on top.
		 */
		augmentation,
		/**
		 * Produce multiple outbound messages - pne per detected face. Also the outbound messages contain the
		 * aligned (e.g. cropped, resized and pre-whitened) image faces in their payloads and the related
		 * annotation metadata in the headers
		 */
		alignment
	}

	/**
	 * Specifies the content type of the outbound message.
	 */
	private OutputMode outputMode = OutputMode.augmentation;

	/**
	 * (for `augmentation` outputMode only!) Draw face annotations on top of the inbound image.
	 */
	private boolean drawFaceAnnotations = true;

	/**
	 * (for `augmentation` outputMode only!) Skip the face annotations from to the outbound message header.
	 */
	private boolean skipFaceAnnotationHeader = false;

	/**
	 * Minimal face size to detect in the input image.
	 */
	private int minFaceSize = 30;

	/**
	 * (Internal config). Control the number of image pyramids used for the face detection
	 */
	private double scaleFactor = 0.709;

	/**
	 * (internal config). Threshold for the first stage of the detection
	 */
	private double stepOneThreshold = 0.6;

	/**
	 * (internal config). Threshold for the second stage of the detection
	 */
	private double stepTwoThreshold = 0.7;

	/**
	 * (internal config). Threshold for the third stage of the detection
	 */
	private double stepThreeThreshold = 0.7;

	/**
	 * Alignment properties. Only applicable for the OutputMode.alignment outputMode!
	 */
	private Alignment alignment = new Alignment();

	public Alignment getAlignment() {
		return alignment;
	}

	public static class Alignment {
		/**
		 * Pre-whitening the aligned face
		 */
		private boolean preWhitening = true;

		/**
		 * Margin for the crop around the bounding box (height, width) in pixels.
		 */
		private int margin = 44;

		/**
		 * Size (height, width) of the aligned face in pixels.
		 */
		private int size = 160;

		public boolean isPreWhitening() {
			return preWhitening;
		}

		public void setPreWhitening(boolean preWhitening) {
			this.preWhitening = preWhitening;
		}

		public int getMargin() {
			return margin;
		}

		public void setMargin(int margin) {
			this.margin = margin;
		}

		public int getSize() {
			return size;
		}

		public void setSize(int size) {
			this.size = size;
		}
	}

	public OutputMode getOutputMode() {
		return outputMode;
	}

	public void setOutputMode(OutputMode outputMode) {
		this.outputMode = outputMode;
	}

	public boolean isDrawFaceAnnotations() {
		return drawFaceAnnotations;
	}

	public void setDrawFaceAnnotations(boolean drawFaceAnnotations) {
		this.drawFaceAnnotations = drawFaceAnnotations;
	}

	public boolean isSkipFaceAnnotationHeader() {
		return skipFaceAnnotationHeader;
	}

	public void setSkipFaceAnnotationHeader(boolean skipFaceAnnotationHeader) {
		this.skipFaceAnnotationHeader = skipFaceAnnotationHeader;
	}

	public int getMinFaceSize() {
		return minFaceSize;
	}

	public void setMinFaceSize(int minFaceSize) {
		this.minFaceSize = minFaceSize;
	}

	public double getScaleFactor() {
		return scaleFactor;
	}

	public void setScaleFactor(double scaleFactor) {
		this.scaleFactor = scaleFactor;
	}

	public double getStepOneThreshold() {
		return stepOneThreshold;
	}

	public void setStepOneThreshold(double stepOneThreshold) {
		this.stepOneThreshold = stepOneThreshold;
	}

	public double getStepTwoThreshold() {
		return stepTwoThreshold;
	}

	public void setStepTwoThreshold(double stepTwoThreshold) {
		this.stepTwoThreshold = stepTwoThreshold;
	}

	public double getStepThreeThreshold() {
		return stepThreeThreshold;
	}

	public void setStepThreeThreshold(double stepThreeThreshold) {
		this.stepThreeThreshold = stepThreeThreshold;
	}
}
