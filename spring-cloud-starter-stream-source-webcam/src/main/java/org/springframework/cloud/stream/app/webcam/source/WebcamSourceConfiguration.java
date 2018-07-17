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

package org.springframework.cloud.stream.app.webcam.source;

import java.awt.Dimension;
import java.time.Duration;
import java.time.temporal.ChronoUnit;

import com.github.sarxos.webcam.Webcam;
import com.github.sarxos.webcam.WebcamUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import reactor.core.publisher.Flux;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.cloud.stream.annotation.EnableBinding;
import org.springframework.cloud.stream.annotation.Output;
import org.springframework.cloud.stream.messaging.Source;
import org.springframework.cloud.stream.reactive.StreamEmitter;
import org.springframework.context.annotation.Bean;
import org.springframework.integration.annotation.InboundChannelAdapter;
import org.springframework.integration.annotation.Poller;
import org.springframework.integration.core.MessageSource;
import org.springframework.messaging.support.GenericMessage;

/**
 *
 * @author Christian Tzolov
 */
@EnableBinding(Source.class)
@EnableConfigurationProperties({ WebcamSourceProperties.class })
public class WebcamSourceConfiguration {

	private static final Log logger = LogFactory.getLog(WebcamSourceConfiguration.class);

	@Autowired
	private WebcamSourceProperties properties;

	@Autowired
	private Webcam webcam;

	@StreamEmitter
	@Output(Source.OUTPUT)
	public Flux<byte[]> emit() {
		return Flux.interval(Duration.of(properties.getCaptureInterval(), ChronoUnit.MILLIS))
				.map(l -> {
					WebcamUtils.capture(webcam, "capturedImage", properties.getImageFormat());
					return WebcamUtils.getImageBytes(webcam, properties.getImageFormat().toLowerCase());
				});
	}

	@Bean
	public Webcam webcam(WebcamSourceProperties properties) {
		Webcam webcam = Webcam.getWebcams().get(properties.getCammeraIndex());
		logger.info("View size" + webcam.getViewSize().width + " : " + webcam.getViewSize().height);
		webcam.setViewSize(new Dimension(properties.getWidth(), properties.getHeight()));
		return webcam;
	}
}
