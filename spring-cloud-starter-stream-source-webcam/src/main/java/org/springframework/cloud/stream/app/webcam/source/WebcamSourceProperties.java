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

package org.springframework.cloud.stream.app.webcam.source;

import com.github.sarxos.webcam.util.ImageUtils;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.validation.annotation.Validated;


/**
 * @author Christian Tzolov
 */
@ConfigurationProperties("webcam")
@Validated
public class WebcamSourceProperties {
	private long captureInterval = 100; //ms

	private String imageFormat = ImageUtils.FORMAT_JPG;

	// W x H: [176x144] [320x240] [640x480]
	private int width = 176;

	private int height = 144;

	/**
	 * When more than one webcams are presented use the camera index specifies which one to use.
	 */
	private int cammeraIndex = 0;

	public long getCaptureInterval() {
		return captureInterval;
	}

	public void setCaptureInterval(long captureInterval) {
		this.captureInterval = captureInterval;
	}

	public String getImageFormat() {
		return imageFormat;
	}

	public void setImageFormat(String imageFormat) {
		this.imageFormat = imageFormat;
	}

	public int getHeight() {
		return height;
	}

	public void setHeight(int height) {
		this.height = height;
	}

	public int getWidth() {
		return width;
	}

	public void setWidth(int width) {
		this.width = width;
	}

	public int getCammeraIndex() {
		return cammeraIndex;
	}

	public void setCammeraIndex(int cammeraIndex) {
		this.cammeraIndex = cammeraIndex;
	}
}
