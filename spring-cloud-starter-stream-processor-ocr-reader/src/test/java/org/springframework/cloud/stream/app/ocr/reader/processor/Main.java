package org.springframework.cloud.stream.app.ocr.reader.processor;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.lept;
import org.bytedeco.javacpp.tesseract;

import static org.bytedeco.javacpp.lept.pixDestroy;
import static org.bytedeco.javacpp.lept.pixRead;

/**
 * @author Christian Tzolov
 */
public class Main {
	// https://github.com/nguyenq/tess4j/wiki/Code-Sample
	public static void main(String[] args) {

		BytePointer outText;

		tesseract.TessBaseAPI api = new tesseract.TessBaseAPI();

		// Initialize tesseract-ocr with English, without specifying tessdata path
		if (api.Init(null, "eng") != 0) {
			System.err.println("Could not initialize tesseract.");
			System.exit(1);
		}

		// Open input image with leptonica library
		lept.PIX image = pixRead("/Users/ctzolov/Dev/projects/scdf/computer-vision/spring-cloud-starter-stream-processor-ocr-reader/src/test/resources/test3.jpg");
		//lept.PIX image = pixRead(args.length > 0 ? args[0] : "/Users/ctzolov/Dev/projects/scdf/computer-vision/spring-cloud-starter-stream-processor-ocr-reader/src/test/resources/test3.jpg");
		api.SetImage(image);
		// Get OCR result
		outText = api.GetUTF8Text();
		System.out.println("OCR output:\n" + outText.getString());

		// Destroy used object and release memory
		api.End();
		outText.deallocate();
		pixDestroy(image);
	}
}
