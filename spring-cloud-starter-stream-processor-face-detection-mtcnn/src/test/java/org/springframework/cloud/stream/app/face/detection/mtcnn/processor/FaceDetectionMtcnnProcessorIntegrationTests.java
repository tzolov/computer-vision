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

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import org.apache.commons.io.IOUtils;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.SpringBootConfiguration;
import org.springframework.boot.autoconfigure.EnableAutoConfiguration;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.cloud.stream.messaging.Processor;
import org.springframework.cloud.stream.test.binder.MessageCollector;
import org.springframework.context.annotation.Import;
import org.springframework.core.io.ClassPathResource;
import org.springframework.integration.support.MessageBuilder;
import org.springframework.messaging.Message;
import org.springframework.test.annotation.DirtiesContext;
import org.springframework.test.context.TestPropertySource;
import org.springframework.test.context.junit4.SpringRunner;
import org.springframework.util.StreamUtils;

import static org.hamcrest.Matchers.equalTo;
import static org.junit.Assert.assertThat;
import static org.springframework.cloud.stream.app.face.detection.mtcnn.processor.FaceDetectionMtcnnProcessorConfiguration.FACE_ANNOTATIONS_HEADER;

/**
 * @author Christian Tzolov
 */
@SuppressWarnings("SpringJavaAutowiringInspection")
@RunWith(SpringRunner.class)
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.NONE)
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_CLASS)
public abstract class FaceDetectionMtcnnProcessorIntegrationTests {

	@Autowired
	protected Processor channels;

	@Autowired
	protected MessageCollector messageCollector;

	@TestPropertySource(properties = { "face.detection.mtcnn.outputMode=annotation" })
	public static class FaceDetectionFaceAnnotationOutputModeTests extends FaceDetectionMtcnnProcessorIntegrationTests {
		@Test
		public void testDetectMultipleFaces() throws IOException {
			try (InputStream is = new ClassPathResource("/VikiMaxiAdi.jpg").getInputStream()) {
				byte[] payload = StreamUtils.copyToByteArray(is);
				channels.input().send(MessageBuilder.withPayload(payload).build());
				Message<String> received = (Message<String>) messageCollector.forChannel(channels.output()).poll();
				Assert.assertNotNull(received);
				String data = received.getPayload();
				assertThat(data, equalTo("[{\"bbox\":{\"x\":331,\"y\":92,\"w\":58,\"h\":71}," +
						"\"confidence\":0.9999871253967285,\"landmarks\":[" +
						"{\"type\":\"LEFT_EYE\",\"position\":{\"x\":346,\"y\":120}}," +
						"{\"type\":\"RIGHT_EYE\",\"position\":{\"x\":374,\"y\":119}}," +
						"{\"type\":\"NOSE\",\"position\":{\"x\":359,\"y\":133}}," +
						"{\"type\":\"MOUTH_LEFT\",\"position\":{\"x\":347,\"y\":147}}," +
						"{\"type\":\"MOUTH_RIGHT\",\"position\":{\"x\":371,\"y\":147}}]}," +
						"{\"bbox\":{\"x\":102,\"y\":159,\"w\":68,\"h\":81},\"confidence\":0.9991292357444763,\"landmarks\":[" +
						"{\"type\":\"LEFT_EYE\",\"position\":{\"x\":121,\"y\":188}}," +
						"{\"type\":\"RIGHT_EYE\",\"position\":{\"x\":153,\"y\":190}}," +
						"{\"type\":\"NOSE\",\"position\":{\"x\":136,\"y\":203}}," +
						"{\"type\":\"MOUTH_LEFT\",\"position\":{\"x\":121,\"y\":218}}," +
						"{\"type\":\"MOUTH_RIGHT\",\"position\":{\"x\":149,\"y\":220}}]}]"));
			}
		}
	}

	@TestPropertySource(properties = { "face.detection.mtcnn.outputMode=augmentation" })
	public static class FaceDetectionAugmentationOutputModeTests extends FaceDetectionMtcnnProcessorIntegrationTests {
		@Test
		public void testDetectMultipleFaces() throws IOException {
			try (InputStream is = new ClassPathResource("/VikiMaxiAdi.jpg").getInputStream()) {
				byte[] payload = StreamUtils.copyToByteArray(is);
				channels.input().send(MessageBuilder.withPayload(payload).build());
				Message<byte[]> received = (Message<byte[]>) messageCollector.forChannel(channels.output()).poll();
				Assert.assertNotNull(received);

				String header = (String) received.getHeaders().get(FACE_ANNOTATIONS_HEADER);
				assertThat(header, equalTo("[{\"bbox\":{\"x\":331,\"y\":92,\"w\":58,\"h\":71}," +
						"\"confidence\":0.9999871253967285,\"landmarks\":[" +
						"{\"type\":\"LEFT_EYE\",\"position\":{\"x\":346,\"y\":120}}," +
						"{\"type\":\"RIGHT_EYE\",\"position\":{\"x\":374,\"y\":119}}," +
						"{\"type\":\"NOSE\",\"position\":{\"x\":359,\"y\":133}}," +
						"{\"type\":\"MOUTH_LEFT\",\"position\":{\"x\":347,\"y\":147}}," +
						"{\"type\":\"MOUTH_RIGHT\",\"position\":{\"x\":371,\"y\":147}}]}," +
						"{\"bbox\":{\"x\":102,\"y\":159,\"w\":68,\"h\":81},\"confidence\":0.9991292357444763,\"landmarks\":[" +
						"{\"type\":\"LEFT_EYE\",\"position\":{\"x\":121,\"y\":188}}," +
						"{\"type\":\"RIGHT_EYE\",\"position\":{\"x\":153,\"y\":190}}," +
						"{\"type\":\"NOSE\",\"position\":{\"x\":136,\"y\":203}}," +
						"{\"type\":\"MOUTH_LEFT\",\"position\":{\"x\":121,\"y\":218}}," +
						"{\"type\":\"MOUTH_RIGHT\",\"position\":{\"x\":149,\"y\":220}}]}]"));

				byte[] data = received.getPayload();
				try (InputStream is2 = new ClassPathResource("/VikiMaxiAdi.jpg_Augmented.jpg").getInputStream()) {
					byte[] augmented = IOUtils.toByteArray(is2);
					assertThat(data, equalTo(augmented));
				}
			}
		}

		@Test
		public void writeAugmentedImages() throws IOException {
			String imageUri = "/VikiMaxiAdi.jpg";
			try (InputStream is = new ClassPathResource(imageUri).getInputStream()) {
				byte[] payload = StreamUtils.copyToByteArray(is);
				channels.input().send(MessageBuilder.withPayload(payload).build());
				Message<byte[]> received = (Message<byte[]>) messageCollector.forChannel(channels.output()).poll();
				byte[] data = received.getPayload();
				IOUtils.write(data, new FileOutputStream("./target/" + imageUri + "_Augmented.jpg"));
			}
		}

	}

	@TestPropertySource(properties = {
			"face.detection.mtcnn.outputMode=alignment",
			"face.detection.mtcnn.alignment.preWhitening=false"
	})
	public static class FaceDetectionMtcnnPayloadTests2 extends FaceDetectionMtcnnProcessorIntegrationTests {

		@Test
		public void testOne() throws IOException {
			try (InputStream is = new ClassPathResource("/webcam.jpg").getInputStream()) {

				byte[] payload = StreamUtils.copyToByteArray(is);

				channels.input().send(MessageBuilder.withPayload(payload).build());

				Message<byte[]> received = (Message<byte[]>) messageCollector.forChannel(channels.output()).poll();

				Assert.assertNotNull(received);

				byte[] data = received.getPayload();

				System.out.println(data.length);

				IOUtils.write(data, new FileOutputStream("./target/test2.jpg"));

			}
		}
	}


	@SpringBootConfiguration
	@EnableAutoConfiguration
	@Import(FaceDetectionMtcnnProcessorConfiguration.class)
	public static class TestFaceDetectionMtcnnProcessorApplication {

	}

}
