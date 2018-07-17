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

package org.springframework.cloud.stream.app.qr.reader.processor;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;

import javax.imageio.ImageIO;

import com.google.zxing.BinaryBitmap;
import com.google.zxing.LuminanceSource;
import com.google.zxing.MultiFormatReader;
import com.google.zxing.NotFoundException;
import com.google.zxing.Result;
import com.google.zxing.client.j2se.BufferedImageLuminanceSource;
import com.google.zxing.common.HybridBinarizer;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import reactor.core.publisher.Flux;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.cloud.stream.annotation.EnableBinding;
import org.springframework.cloud.stream.annotation.Input;
import org.springframework.cloud.stream.annotation.StreamListener;
import org.springframework.cloud.stream.messaging.Processor;
import org.springframework.messaging.handler.annotation.SendTo;

/**
 *
 * @author Christian Tzolov
 */
@EnableBinding(Processor.class)
@EnableConfigurationProperties({ QrReaderProcessorProperties.class })
public class QrReaderProcessorConfiguration {

	private static final Log logger = LogFactory.getLog(QrReaderProcessorConfiguration.class);

	@Autowired
	private QrReaderProcessorProperties properties;

    @StreamListener
    @SendTo(Processor.OUTPUT)
	public Flux<String> evaluate(@Input(Processor.INPUT) Flux<byte[]> imageInBytes) {
		return imageInBytes.map(imgBytes -> decodeQrImage(imgBytes));
	}

	private String decodeQrImage(byte[] imageBytes) {
		try (InputStream in = new ByteArrayInputStream(imageBytes)) {
			BufferedImage bufferedImage = ImageIO.read(in);
			LuminanceSource source = new BufferedImageLuminanceSource(bufferedImage);
			BinaryBitmap bitmap = new BinaryBitmap(new HybridBinarizer(source));

			Result result = new MultiFormatReader().decodeWithState(bitmap);

			if (result != null) {
				return result.getText();
			}
		}
		catch (NotFoundException e) {
			// No QR found in the input image. Do Nothing!
		}
		catch (IOException e) {
			logger.warn("QR decoding failed to read the input byte array!", e);
		}

		// Null mend that QR image is found and not output message will be send.
		return null;
	}

}
