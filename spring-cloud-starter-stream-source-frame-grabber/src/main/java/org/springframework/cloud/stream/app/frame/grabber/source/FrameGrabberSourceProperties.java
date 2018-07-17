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

package org.springframework.cloud.stream.app.frame.grabber.source;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.core.io.Resource;
import org.springframework.validation.annotation.Validated;


/**
 * @author Christian Tzolov
 */
@ConfigurationProperties("frame.grabber")
@Validated
public class FrameGrabberSourceProperties {

	public enum ImageType {
		RGB(1),
		ARGB(2),
		ARGB_PRE(3),
		BRG(4),
		BRG_3BYTE(5),
		BRG_4BYTE(6),
		BRG_PRE_4BYTE(7),
		USHORT_565_RGB(8),
		USHORT_555_RGB(9),
		BYTE_GRAY(10),
		USHORT_GRAY(11),
		BYTE_BINARY(12),
		BYTE_INDEXED(13);

		public final int typeId;

		ImageType(int typeId) {
			this.typeId = typeId;
		}
	}

	/**
	 * Image type send as byte array
	 */
	private ImageType outputImageType = ImageType.BRG;

	/**
	 * Capture interval
	 */
	private long captureInterval = 100; //ms


	/**
	 * Capture width
	 */
	private int width = 176;

	/**
	 * Capture height
	 */
	private int height = 144;

	/**
	 * When more than one webcams are presented use the camera index specifies which one to use.
	 */
	private int deviceIndex = 0;

	/**
	 * Location of an input video file to grab frames from. Supports file:/ , http:// or , classpath:/ locations.
	 * If null the grabber will use the Webcams as input video stream
	 */
	private Resource inputVideoUri = null;

	public long getCaptureInterval() {
		return captureInterval;
	}

	public void setCaptureInterval(long captureInterval) {
		this.captureInterval = captureInterval;
	}

	public int getWidth() {
		return width;
	}

	public void setWidth(int width) {
		this.width = width;
	}

	public int getHeight() {
		return height;
	}

	public void setHeight(int height) {
		this.height = height;
	}

	public int getDeviceIndex() {
		return deviceIndex;
	}

	public void setDeviceIndex(int deviceIndex) {
		this.deviceIndex = deviceIndex;
	}

	public ImageType getOutputImageType() {
		return outputImageType;
	}

	public void setOutputImageType(ImageType outputImageType) {
		this.outputImageType = outputImageType;
	}

	public Resource getInputVideoUri() {
		return inputVideoUri;
	}

	public void setInputVideoUri(Resource inputVideoUri) {
		this.inputVideoUri = inputVideoUri;
	}
}
