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

package org.springframework.cloud.stream.app.face.recognition.processor;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import javax.imageio.ImageIO;

import org.apache.commons.io.IOUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.cloud.stream.annotation.EnableBinding;
import org.springframework.cloud.stream.app.tensorflow.processor.TensorFlowService;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.integration.support.MessageBuilder;
import org.springframework.messaging.Message;
import org.springframework.cloud.stream.annotation.StreamListener;
import org.springframework.messaging.MessageHeaders;
import org.springframework.messaging.handler.annotation.SendTo;
import org.springframework.cloud.stream.messaging.Processor;
import org.springframework.util.Assert;
import org.springframework.util.CollectionUtils;
import org.springframework.util.MimeTypeUtils;

/**
 *
 * @author Christian Tzolov
 */
@Configuration
@EnableBinding(Processor.class)
@EnableConfigurationProperties({ FaceRecognitionProcessorProperties.class })
public class FaceRecognitionProcessorConfiguration {

	private static final Log logger = LogFactory.getLog(FaceRecognitionProcessorConfiguration.class);

	@Autowired
	private FaceRecognitionProcessorProperties properties;

	@Autowired
	private FaceRecognitionService faceRecognitionService;

	@Bean
	public FaceRecognitionService faceRecognitionService() throws IOException {
		TensorFlowService tfService = new TensorFlowService(this.properties.getModel());
		return new FaceRecognitionService(tfService);
	}

	public static class FaceEmbeddingRegistry extends HashMap<String, float[]> {
	}

	@Autowired
	private FaceEmbeddingRegistry faceEmbeddings;

	@Bean
	public FaceEmbeddingRegistry faceEmbeddings()  {
		FaceEmbeddingRegistry faceEmbeddings = new FaceEmbeddingRegistry();

		//
		if (properties.getEmbeddingDirectory() != null) {
			Assert.isTrue(properties.getEmbeddingDirectory().isDirectory(),
					"Not a directory" + properties.getEmbeddingDirectory().getAbsolutePath());

			for (File file : properties.getEmbeddingDirectory().listFiles()) {
				try {
					byte[] image = IOUtils.toByteArray(file.toURI());
					float[] faceFingerpring = faceRecognitionService.encode(toBufferedImage(image));
					faceEmbeddings.put(file.getName(), faceFingerpring);
				}
				catch (IOException ioe) {
					logger.error("Failed to load face embedding", ioe);
				}
			}
		}

		if (!CollectionUtils.isEmpty(properties.getMap())) {
			for (Map.Entry<String, float[]> e : properties.getMap().entrySet()) {
				faceEmbeddings.put(e.getKey(), e.getValue());
			}
		}

		return faceEmbeddings;
	}

	@StreamListener(Processor.INPUT)
    @SendTo(Processor.OUTPUT)
    public Object evaluate(Message<?> input) {
		if (!(input.getPayload() instanceof byte[])) {
			throw new IllegalArgumentException(String.format("Expected byte[] json type, found: %s",
					input.getPayload().getClass().getCanonicalName()));
		}
		try {
			byte[] inputImageAsBytes = (byte[]) input.getPayload();
			float[] faceFingerpring = faceRecognitionService.encode(toBufferedImage(inputImageAsBytes));

			double closestDistance = Double.MAX_VALUE;
			String closestDistanceName = "NONE";
			for (Map.Entry<String, float[]> e : this.faceEmbeddings.entrySet()) {
				double d = distance(faceFingerpring, e.getValue());
				if (d < closestDistance) {
					closestDistance = d;
					closestDistanceName = e.getKey();
				}
			}
			return MessageBuilder
					.withPayload(closestDistanceName + ":" + closestDistance)
					.setHeader(MessageHeaders.CONTENT_TYPE, MimeTypeUtils.TEXT_PLAIN)
					.build();
		}
		catch (IOException e) {
			logger.error("", e);
			return "ERROR";
		}
	}

	private  double distance(float[] emb1, float[] emb2) {
		Assert.isTrue(emb1.length == emb2.length, "Compared arrays must be of same size, but " +
				"emb1: " + emb1.length + " and emb2: " + emb2.length);
		double sum = 0;
		for (int i = 0; i < emb1.length; i++) {
			sum = (emb1[i] - emb2[i]) * (emb1[i] - emb2[i]);
		}
		return Math.sqrt(sum);
	}

	private BufferedImage toBufferedImage(byte[] image) throws IOException {
		try (ByteArrayInputStream is = new ByteArrayInputStream(image)) {
			return ImageIO.read(is);
		}
	}
}
