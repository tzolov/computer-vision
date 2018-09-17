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

import java.io.File;
import java.util.HashMap;
import java.util.Map;

import javax.validation.constraints.NotNull;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.core.io.Resource;
import org.springframework.validation.annotation.Validated;


/**
 * @author Christian Tzolov
 */
@ConfigurationProperties("face.recognition")
@Validated
public class FaceRecognitionProcessorProperties {

	@NotNull
	private Resource model;

	private File embeddingDirectory;

	private Map<String, float[]> map = new HashMap<>();

	public Resource getModel() {
		return model;
	}

	public void setModel(Resource model) {
		this.model = model;
	}

	public Map<String, float[]> getMap() {
		return map;
	}

	public File getEmbeddingDirectory() {
		return embeddingDirectory;
	}

	public void setEmbeddingDirectory(File embeddingDirectory) {
		this.embeddingDirectory = embeddingDirectory;
	}
}
