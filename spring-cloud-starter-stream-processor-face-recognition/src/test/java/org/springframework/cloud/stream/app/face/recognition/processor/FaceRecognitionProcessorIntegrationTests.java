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

import java.io.IOException;
import java.io.InputStream;

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
import org.springframework.core.io.DefaultResourceLoader;
import org.springframework.integration.support.MessageBuilder;
import org.springframework.messaging.Message;
import org.springframework.test.annotation.DirtiesContext;
import org.springframework.test.context.TestPropertySource;
import org.springframework.test.context.junit4.SpringRunner;
import org.springframework.util.StreamUtils;

/**
 * @author Christian Tzolov
 */
@SuppressWarnings("SpringJavaAutowiringInspection")
@RunWith(SpringRunner.class)
@SpringBootTest(
		webEnvironment = SpringBootTest.WebEnvironment.NONE,
		properties = {}
		)
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_CLASS)
public abstract class FaceRecognitionProcessorIntegrationTests {

	@Autowired
	protected Processor channels;

	@Autowired
	protected MessageCollector messageCollector;

	//@TestPropertySource(
	//		//properties = { "debug=true", "logging.level.*=DEBUG", }
	//		)
	public static class FaceRecognitionPayloadTests extends FaceRecognitionProcessorIntegrationTests {

		@Test
		public void testOne() throws IOException {

			try (InputStream is = new DefaultResourceLoader().getResource("file:/Users/ctzolov/Dev/projects/scdf/computer-vision/spring-cloud-starter-stream-processor-face-recognition/src/test/resources/cropped0.png").getInputStream()) {
			//try (InputStream is = new DefaultResourceLoader().getResource("file:/Users/ctzolov/Dev/projects/mtcnn/target/cropped0.png").getInputStream()) {

				byte[] payload = StreamUtils.copyToByteArray(is);

				channels.input().send(MessageBuilder.withPayload(payload).build());

				Message<?> received = messageCollector.forChannel(channels.output()).poll();

				System.out.println(received.getPayload());

				Assert.assertNotNull(received);
			}
		}
	}

	@SpringBootConfiguration
	@EnableAutoConfiguration
	@Import(FaceRecognitionProcessorConfiguration.class)
	public static class TestFaceRecognitionProcessorApplication {

	}

}
