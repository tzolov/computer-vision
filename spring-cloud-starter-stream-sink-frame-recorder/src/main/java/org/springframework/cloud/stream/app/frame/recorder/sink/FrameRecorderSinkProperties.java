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

package org.springframework.cloud.stream.app.frame.recorder.sink;

import java.io.File;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.validation.annotation.Validated;


/**
 * @author Christian Tzolov
 */
@ConfigurationProperties("frame.recorder.sink")
@Validated
public class FrameRecorderSinkProperties {

	private static final String DEFAULT_DIR = System.getProperty("java.io.tmpdir") +
			File.separator + "frame-recorder-sink" + File.separator + "output";

	private static final String DEFAULT_NAME = "frame-recorder-sink.mp4";

	/**
	 * Output video file location
	 */
	private File outputFile;

	/**
	 * Recorded image width
	 */
	private int width = 640;

	/**
	 * Recorded image height
	 */
	private int height = 480;

	/**
	 * Set the Video format (e.g. mp4, flv, avi ...) explicitly. If empty Recorder will auto detect the output format from the name.
	 */
	private String videoFormat;

	/**
	 * 0 - no Audio supported
	 */
	private int audioChannel = 0;

	public File getOutputFile() {

		return outputFile;
	}

	public void setOutputFile(File outputFile) {
		this.outputFile = outputFile;
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

	public int getAudioChannel() {
		return audioChannel;
	}

	public void setAudioChannel(int audioChannel) {
		this.audioChannel = audioChannel;
	}

	public String getVideoFormat() {
		return videoFormat;
	}

	public void setVideoFormat(String videoFormat) {
		this.videoFormat = videoFormat;
	}
}
