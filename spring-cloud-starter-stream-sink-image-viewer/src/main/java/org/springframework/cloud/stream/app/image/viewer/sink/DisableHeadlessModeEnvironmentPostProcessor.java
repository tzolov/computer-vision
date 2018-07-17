package org.springframework.cloud.stream.app.image.viewer.sink;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.env.EnvironmentPostProcessor;
import org.springframework.core.env.ConfigurableEnvironment;

/**
 * @author Christian Tzolov
 */
public class DisableHeadlessModeEnvironmentPostProcessor implements EnvironmentPostProcessor {
	private static final String SYSTEM_PROPERTY_JAVA_AWT_HEADLESS = "java.awt.headless";
	@Override
	public void postProcessEnvironment(ConfigurableEnvironment environment, SpringApplication application) {
		System.setProperty(SYSTEM_PROPERTY_JAVA_AWT_HEADLESS, Boolean.toString(false));
	}
}
