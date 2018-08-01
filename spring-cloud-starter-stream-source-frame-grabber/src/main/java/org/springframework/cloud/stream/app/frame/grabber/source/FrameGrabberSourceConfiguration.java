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

package org.springframework.cloud.stream.app.frame.grabber.source;

import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.time.Duration;
import java.time.temporal.ChronoUnit;

import javax.imageio.ImageIO;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.Java2DFrameUtils;
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
import org.springframework.messaging.Message;
import org.springframework.messaging.support.GenericMessage;
import org.springframework.messaging.support.MessageBuilder;

/**
 *
 * @author Christian Tzolov
 */
@EnableBinding(Source.class)
@EnableConfigurationProperties({ FrameGrabberSourceProperties.class })
public class FrameGrabberSourceConfiguration {

	private static final Log logger = LogFactory.getLog(FrameGrabberSourceConfiguration.class);

	@Autowired
	private FrameGrabberSourceProperties properties;

	@Autowired
	private FrameGrabber frameGrabber;

	//@StreamEmitter
	//@Output(Source.OUTPUT)
	//public Flux<byte[]> emit() {
	//	return Flux.interval(Duration.of(this.properties.getCaptureInterval(), ChronoUnit.MILLIS))
	//			.map(l -> {
	//				try {
	//					Frame frame = this.frameGrabber.grab();
	//					BufferedImage image = Java2DFrameUtils.toBufferedImage(frame);
	//					return imageToBytes(resize(image, properties.getWidth(), properties.getHeight()));
	//				}
	//				catch (Exception e) {
	//					logger.error("Failed to grab the frame or to convert the image", e);
	//				}
	//				return null;
	//			});
	//}

	@Bean
	@InboundChannelAdapter(value = Source.OUTPUT, poller = @Poller(fixedDelay = "${frame.grabber.captureInterval:1000}", maxMessagesPerPoll = "1"))
	public MessageSource<byte[]> myMessageSource() {
		return () -> {
			try {
				Frame frame = frameGrabber.grab();
				BufferedImage image = Java2DFrameUtils.toBufferedImage(frame);
				return MessageBuilder.withPayload(
						imageToBytes(resize(image, properties.getWidth(), properties.getHeight()))).build();
			}
			catch (Exception e) {
				logger.error("Failed to grab the frame or to convert the image", e);
			}
			return null;

		};
	}

	private byte[] imageToBytes(BufferedImage bufferedImage) throws IOException {
		ByteArrayOutputStream baos = new ByteArrayOutputStream();
		ImageIO.write(bufferedImage, "png", baos);
		baos.flush();
		byte[] imageInBytes = baos.toByteArray();
		baos.close();
		return imageInBytes;
	}

	private BufferedImage resize(BufferedImage originalImage, int newWidth, int newHeight) {
		Image tmpImage = originalImage.getScaledInstance(newWidth, newHeight, Image.SCALE_SMOOTH);
		BufferedImage resizedImage = new BufferedImage(newWidth, newHeight, properties.getOutputImageType().typeId);

		Graphics2D g2d = resizedImage.createGraphics();
		g2d.drawImage(tmpImage, 0, 0, null);
		g2d.dispose();

		return resizedImage;
	}

	@Bean
	public FrameGrabber frameGrabber(FrameGrabberSourceProperties properties) throws IOException {
		FrameGrabber grabber;
		if (properties.getInputVideoUri() != null) {
			grabber = new FFmpegFrameGrabber(properties.getInputVideoUri().getInputStream());
		}
		else {
			grabber = FrameGrabber.createDefault(properties.getDeviceIndex());
		}
		grabber.setImageWidth(properties.getWidth());
		grabber.setImageHeight(properties.getHeight());
		grabber.start();
		logger.info("Frame Grabber Started: " + grabber.toString());
		return grabber;
	}
}
