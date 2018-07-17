/*
 * Copyright 2017--2018 the original author or authors.
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

package org.springframework.cloud.stream.app.ocr.reader.processor;

import javax.annotation.PreDestroy;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.lept;
import org.bytedeco.javacpp.tesseract;
import reactor.core.publisher.Flux;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.cloud.stream.annotation.EnableBinding;
import org.springframework.cloud.stream.annotation.Input;
import org.springframework.cloud.stream.annotation.StreamListener;
import org.springframework.cloud.stream.messaging.Processor;
import org.springframework.context.annotation.Bean;
import org.springframework.messaging.Message;
import org.springframework.messaging.handler.annotation.SendTo;

import static org.bytedeco.javacpp.lept.pixDestroy;
import static org.bytedeco.javacpp.lept.pixRead;
import static org.bytedeco.javacpp.lept.pixReadMem;

/**
 *
 * @author Christian Tzolov
 */
@EnableBinding(Processor.class)
@EnableConfigurationProperties({ OcrReaderProcessorProperties.class })
public class OcrReaderProcessorConfiguration {

	private static final Log logger = LogFactory.getLog(OcrReaderProcessorConfiguration.class);

	@Autowired
	private OcrReaderProcessorProperties properties;

	@Autowired
	private tesseract.TessBaseAPI tessBaseApi;

	@StreamListener
	@SendTo(Processor.OUTPUT)
	public Flux<String> evaluate(@Input(Processor.INPUT) Flux<byte[]> imageInBytes) {
		return imageInBytes.map(imgBytes -> process(imgBytes));
	}

	private String process(byte[] byteImage) {
		BytePointer outText;
		lept.PIX image = pixReadMem(byteImage, byteImage.length);
		this.tessBaseApi.SetImage(image);
		// Get OCR result
		outText = this.tessBaseApi.GetUTF8Text();
		String text = outText.getString();
		outText.deallocate();
		pixDestroy(image);
		return text;
	}

	@Bean
	public tesseract.TessBaseAPI tessBaseAPI() {
		tesseract.TessBaseAPI api = new tesseract.TessBaseAPI();

		// Initialize tesseract-ocr with English, without specifying tessdata path
		//if (api.Init("BOOT-INF/classes/", "eng") != 0) {
		if (api.Init(null, "eng") != 0) {
			throw new IllegalArgumentException("Could not initialize tesseract.");
		}
		return api;
	}

	@PreDestroy
	public void de() {
		tessBaseApi.End();
		//		outText.deallocate();
		//		pixDestroy(image);
	}
}
