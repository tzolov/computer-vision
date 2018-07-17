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

package org.springframework.cloud.stream.app.frame.recorder.sink;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;

import javax.annotation.PreDestroy;
import javax.imageio.ImageIO;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.bytedeco.javacv.FFmpegFrameRecorder;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameRecorder;
import org.bytedeco.javacv.Java2DFrameUtils;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.cloud.stream.annotation.EnableBinding;
import org.springframework.cloud.stream.messaging.Sink;
import org.springframework.context.annotation.Bean;
import org.springframework.integration.annotation.ServiceActivator;
import org.springframework.messaging.Message;
import org.springframework.util.StringUtils;

/**
 *
 * @author Christian Tzolov
 */
@EnableBinding(Sink.class)
@EnableConfigurationProperties({ FrameRecorderSinkProperties.class })
public class FrameRecorderSinkConfiguration {

	private static final Log logger = LogFactory.getLog(FrameRecorderSinkConfiguration.class);

	@Autowired
	private FrameRecorderSinkProperties properties;

	@Autowired
	private FrameRecorder frameRecorder;

	@ServiceActivator(inputChannel = Sink.INPUT)
	public void handle(Message<?> message) {

		byte[] imageBytes = (byte[]) message.getPayload();
		try {
			BufferedImage bImage = ImageIO.read(new ByteArrayInputStream(imageBytes));
			Frame frame = Java2DFrameUtils.toFrame(bImage);
			frameRecorder.record(frame);
		}
		catch (IOException e) {
			logger.error(e);
		}
	}

	@Bean
	public FrameRecorder frameRecorder() throws FrameRecorder.Exception {

		FFmpegFrameRecorder recorder = new FFmpegFrameRecorder(this.properties.getOutputFile(),
				this.properties.getWidth(), this.properties.getHeight(), this.properties.getAudioChannel());

		// If the video format is not set explicitly, the FFmpegFrameRecorder will try to detect it from the name
		if (StringUtils.hasText(this.properties.getVideoFormat())) {
			recorder.setFormat(this.properties.getVideoFormat());
		}
		recorder.start();
		return recorder;
	}

	@PreDestroy
	public void preDestroy() throws FrameRecorder.Exception {
		this.frameRecorder.stop();
	}
}
