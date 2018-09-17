package org.springframework.cloud.stream.app.face.detection.mtcnn.processor.attic;

import java.io.IOException;

import com.fasterxml.jackson.databind.ObjectMapper;
import net.tzolov.cv.mtcnn.FaceAnnotation;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;

import static org.hamcrest.Matchers.equalTo;
import static org.junit.Assert.assertThat;

/**
 * @author Christian Tzolov
 */
public class MtcnnServiceJava2DTest {

	private MtcnnServiceJava2D mtcnnService;

	@Before
	public void before() throws IOException {
		mtcnnService = new MtcnnServiceJava2D();
	}

	@Ignore
	@Test
	public void testSingeFace() throws IOException {

		FaceAnnotation[] boundingBoxes = mtcnnService.faceDetection("classpath:/ivan.jpg");
		String bboxJson = new ObjectMapper().writeValueAsString(boundingBoxes);

		assertThat(bboxJson, equalTo("[{\"box\":[278,90,48,64],\"confidence\":0.9955518841743469,\"keypoints\":{\"left_eye\":[291,117],\"right_eye\":[314,114],\"nose\":[303,131],\"mouth_left\":[296,143],\"mouth_right\":[313,141]}}]"));
	}

	@Ignore
	@Test
	public void test3Faces() throws IOException {

		FaceAnnotation[] boundingBoxes = mtcnnService.faceDetection("classpath:/VikiMaxiAdi.jpg");
		String bboxJson = new ObjectMapper().writeValueAsString(boundingBoxes);

		assertThat(bboxJson, equalTo("[{\"box\":[332,92,56,71],\"confidence\":0.9999566078186035,\"keypoints\":{\"left_eye\":[347,121],\"right_eye\":[373,120],\"nose\":[358,133],\"mouth_left\":[347,146],\"mouth_right\":[370,146]}},{\"box\":[102,154,68,85],\"confidence\":0.9984952211380005,\"keypoints\":{\"left_eye\":[122,189],\"right_eye\":[153,190],\"nose\":[136,203],\"mouth_left\":[122,218],\"mouth_right\":[150,220]}}]"));

	}
}
