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

package org.springframework.cloud.stream.app.computer.vision.common.segmentation;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.validation.annotation.Validated;


/**
 * @author Christian Tzolov
 */
@ConfigurationProperties("semantic.segmentation")
@Validated
public class SemanticSegmentationProcessorProperties {

	/**
	 * Blended mask transparency. Value is between 0.0 (0% transparency) and 1.0 (100% transparent).
	 */
	public double maskTransparency = 0.3;

	/**
	 * When true the semantic segmentation masks are blend with the input image.
	 */
	public boolean maskBlendingEnabled = true;

	public boolean isMaskBlendingEnabled() {
		return maskBlendingEnabled;
	}

	public void setMaskBlendingEnabled(boolean maskBlendingEnabled) {
		this.maskBlendingEnabled = maskBlendingEnabled;
	}

	public double getMaskTransparency() {
		return maskTransparency;
	}

	public void setMaskTransparency(double maskTransparency) {
		this.maskTransparency = maskTransparency;
	}
}
