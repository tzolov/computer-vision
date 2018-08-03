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

package org.springframework.cloud.stream.app.semantic.segmentation.processor;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import javax.imageio.ImageIO;

import org.apache.commons.io.IOUtils;
import org.junit.Assert;
import org.junit.Ignore;
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

/**
 * @author Christian Tzolov
 */
@SuppressWarnings("SpringJavaAutowiringInspection")
@RunWith(SpringRunner.class)
@SpringBootTest(
		webEnvironment = SpringBootTest.WebEnvironment.NONE,
		properties = {
				//"debug=false",
		})
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_CLASS)
public abstract class SemanticSegmentationProcessorIntegrationTests {

	@Autowired
	protected Processor channels;

	@Autowired
	protected MessageCollector messageCollector;

	@TestPropertySource(properties = {
			"tensorflow.mode=header",
			"tensorflow.modelFetch=SemanticPredictions:0",
			"tensorflow.model=file:/Users/ctzolov/Downloads/deeplabv3_mnv2_pascal_train_aug/frozen_inference_graph.pb",

	})
	public static class SemanticSegmentationPayloadTests extends SemanticSegmentationProcessorIntegrationTests {

		@Test
		public void testOne() throws IOException {
			try (InputStream is = new ClassPathResource("/images/VikiMaxiAdi.jpg").getInputStream()) {

				byte[] payload = StreamUtils.copyToByteArray(is);

				channels.input().send(MessageBuilder.withPayload(payload).build());

				Message<byte[]> received = (Message<byte[]>) messageCollector.forChannel(channels.output()).poll();

				Assert.assertNotNull(received);

				byte[] data = received.getPayload();

				System.out.println(data.length);

				IOUtils.write(data, new FileOutputStream("./target/test1.jpg"));
			}
		}
	}

	@SpringBootConfiguration
	@EnableAutoConfiguration
	@Import(SemanticSegmentationProcessorConfiguration.class)
	public static class TestSemanticSegmentationProcessorApplication {

	}

}
