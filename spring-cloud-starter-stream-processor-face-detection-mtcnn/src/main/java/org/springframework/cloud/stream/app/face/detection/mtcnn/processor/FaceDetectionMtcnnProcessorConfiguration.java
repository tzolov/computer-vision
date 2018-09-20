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

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.util.concurrent.atomic.AtomicInteger;

import javax.imageio.ImageIO;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import net.tzolov.cv.mtcnn.FaceAnnotation;
import net.tzolov.cv.mtcnn.MtcnnService;
import net.tzolov.cv.mtcnn.MtcnnUtil;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.datavec.image.loader.Java2DNativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.cloud.stream.annotation.EnableBinding;
import org.springframework.cloud.stream.annotation.StreamListener;
import org.springframework.cloud.stream.messaging.Processor;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.integration.support.MessageBuilder;
import org.springframework.messaging.Message;
import org.springframework.messaging.MessageChannel;
import org.springframework.messaging.MessageHeaders;
import org.springframework.util.Assert;
import org.springframework.util.MimeTypeUtils;

/**
 *
 * @author Christian Tzolov
 */
@Configuration
@EnableBinding(Processor.class)
@EnableConfigurationProperties({ FaceDetectionMtcnnProcessorProperties.class })
public class FaceDetectionMtcnnProcessorConfiguration {

	private static final Log logger = LogFactory.getLog(FaceDetectionMtcnnProcessorConfiguration.class);
	public static final String FACE_ANNOTATIONS_HEADER = "faceAnnotations";
	public static final String FACE_ANNOTATION_HEADER = "annotation";

	private ObjectMapper mapper = new ObjectMapper();

	private Java2DNativeImageLoader loader = new Java2DNativeImageLoader();

	@Autowired
	private FaceDetectionMtcnnProcessorProperties properties;

	@Autowired
	private MtcnnService mtcnnService;

	@Autowired
	private MessageChannel output;

	@Bean
	public MtcnnService mtcnnService() {
		return new MtcnnService(properties.getMinFaceSize(), properties.getScaleFactor(), new double[] {
				properties.getStepOneThreshold(), properties.getStepTwoThreshold(), properties.getStepThreeThreshold() });
	}

	@StreamListener(Processor.INPUT)
	public void evaluate(Message<?> input) {

		if (!(input.getPayload() instanceof byte[])) {
			throw new IllegalArgumentException(String.format("Expected byte[] json type, found: %s",
					input.getPayload().getClass().getCanonicalName()));
		}

		try {
			byte[] inputImageAsBytes = (byte[]) input.getPayload();

			FaceAnnotation[] faceAnnotations = this.mtcnnService.faceDetection(inputImageAsBytes);

			switch (this.properties.getOutputMode()) {
			case annotation:
				// Use the json json to send the detected bounding boxes downstream
				this.output.send(MessageBuilder.fromMessage(input)
						.withPayload(toJson(faceAnnotations))
						.setHeader(MessageHeaders.CONTENT_TYPE, MimeTypeUtils.APPLICATION_JSON)
						.build());
				break;

			case augmentation:
				// Sends the original augmentation (possible augmented with the detected faces) inside the payload while the
				// detected bounding boxes are passed as json header called boundingBoxes
				byte[] outImage = this.properties.isDrawFaceAnnotations() ?
						drawFaceAnnotations(inputImageAsBytes, faceAnnotations) : inputImageAsBytes;
				MessageBuilder mb = MessageBuilder.fromMessage(input)
						.withPayload(outImage)
						.setHeader(MessageHeaders.CONTENT_TYPE, MimeTypeUtils.APPLICATION_OCTET_STREAM);
				if (!this.properties.isSkipFaceAnnotationHeader()) {
					mb.setHeader(FACE_ANNOTATIONS_HEADER, toJson(faceAnnotations));
				}
				this.output.send(mb.build());
				break;

			case alignment:
				// Generate multiple outbound messages - pne per detected face. Also the outbound messages contain the
				// aligned (e.g. cropped, resized and pre-whitened) image faces in their payloads and the related
				// annotation metadata in the headers.
				INDArray inputImageAsNdArray = toNDArray(inputImageAsBytes);

				INDArray[] alignedFaces = this.mtcnnService.faceAlignment(
						inputImageAsNdArray, faceAnnotations, this.properties.getAlignment().getMargin(),
						this.properties.getAlignment().getSize(), this.properties.getAlignment().isPreWhitening());

				Assert.isTrue(alignedFaces.length == faceAnnotations.length, "The number of aligned images " +
						"must match the number of face detected bounding boxes");

				for (int i = 0; i < alignedFaces.length; i++) {
					BufferedImage alignedFaceBufferedImage = this.loader.asBufferedImage(alignedFaces[i]);
					byte[] alignedFaceAsBytes = MtcnnUtil.toByteArray(alignedFaceBufferedImage, "png");
					this.output.send(MessageBuilder
							.withPayload(alignedFaceAsBytes).fromMessage(input)
							.setHeader(MessageHeaders.CONTENT_TYPE, MimeTypeUtils.APPLICATION_OCTET_STREAM)
							.setHeader(FACE_ANNOTATION_HEADER, toJson(faceAnnotations[i]))
							.build());
				}
				break;
			}
		}
		catch (Exception e) {
			logger.error("Failed to process the input", e);
			if (logger.isDebugEnabled()) {
				debugSaveImage((byte[]) input.getPayload());
			}
			output.send(input); // In case of exception send the origin augmentation forward.
		}
	}

	private BufferedImage toBufferedImage(byte[] image) throws IOException {
		try (ByteArrayInputStream is = new ByteArrayInputStream(image)) {
			return ImageIO.read(is);
		}
	}

	private INDArray toNDArray(byte[] image) throws IOException {
		return this.loader.asMatrix(toBufferedImage(image))
				.get(new INDArrayIndex[] {
						NDArrayIndex.point(0L), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all() });
	}

	private byte[] drawFaceAnnotations(byte[] inputImage, FaceAnnotation[] faceAnnotations) throws IOException {
		BufferedImage originalImage = toBufferedImage(inputImage);
		originalImage = MtcnnUtil.drawFaceAnnotations(originalImage, faceAnnotations);
		return MtcnnUtil.toByteArray(originalImage, "png");
	}

	private String toJson(Object o) throws JsonProcessingException {
		return this.mapper.writeValueAsString(o);
	}

	private AtomicInteger debugErrorCounter = new AtomicInteger(0);

	/**
	 * Utility that allows to save the input augmentation that may have caused an exception.
	 * @param byteImage augmentation to save
	 */
	private void debugSaveImage(byte[] byteImage) {
		try {
			ByteArrayInputStream is = new ByteArrayInputStream(byteImage);
			BufferedImage bufferedImage = ImageIO.read(is);
			ImageIO.write(bufferedImage, "png", new File(
					"target/error_" + debugErrorCounter.getAndIncrement() + ".png"));
		}
		catch (IOException e) {
			e.printStackTrace();
		}
	}
}
