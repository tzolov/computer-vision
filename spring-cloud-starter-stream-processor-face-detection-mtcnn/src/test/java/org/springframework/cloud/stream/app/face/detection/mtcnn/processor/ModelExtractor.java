package org.springframework.cloud.stream.app.face.detection.mtcnn.processor;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Optional;

import org.apache.commons.compress.archivers.ArchiveEntry;
import org.apache.commons.compress.archivers.ArchiveInputStream;
import org.apache.commons.compress.archivers.ArchiveStreamFactory;
import org.apache.commons.compress.compressors.CompressorInputStream;
import org.apache.commons.compress.compressors.CompressorStreamFactory;

import org.springframework.core.io.DefaultResourceLoader;
import org.springframework.core.io.Resource;
import org.springframework.util.StreamUtils;
import org.springframework.util.StringUtils;

/**
 * Extracts a pre-trained (frozen) Tensorflow model URI into byte array. The 'http://', 'file://' and 'classpath://'
 * URI schemas are supported.
 *
 * Models can be extract either from raw files or form compressed archives. When  extracted from an archive the model
 * file name can optionally be provided as an URI fragment. For example for resource: http://myarchive.tar.gz#model.pb
 * the myarchive.tar.gz is traversed to uncompress and extract the model.pb file as byte array.
 * If the file name is not provided as URI fragment then the first file in the archive with extension .pb is extracted.
 *
 * @author Christian Tzolov
 */
public class ModelExtractor {

	private static final String DEFAULT_FROZEN_GRAPH_FILE_EXTENSION = ".pb";

	/**
	 * When an archive resource if referred, but no fragment URI is provided (to specify the target file name in
	 * the archive) then the extractor selects the first file in the archive with the extension that match
	 * the frozenGraphFileExtension (defaults to .pb).
	 */
	public final String frozenGraphFileExtension;

	public ModelExtractor() {
		this(DEFAULT_FROZEN_GRAPH_FILE_EXTENSION);
	}

	public ModelExtractor(String frozenGraphFileExtension) {
		this.frozenGraphFileExtension = frozenGraphFileExtension;
	}

	public byte[] getModel(Resource modelResource) throws Exception {

		String[] archiveCompressor = detectArchiveAndCompressor(modelResource.getFilename());
		String archive = archiveCompressor[0];
		String compressor = archiveCompressor[1];
		String fragment = modelResource.getURI().getFragment();

		try (InputStream is = modelResource.getInputStream();
			 InputStream bi = new BufferedInputStream(is)) {

			if (StringUtils.hasText(compressor)) {
				try (CompressorInputStream cis = new CompressorStreamFactory().createCompressorInputStream(compressor, bi)) {
					if (StringUtils.hasText(archive)) {
						try (ArchiveInputStream ais = new ArchiveStreamFactory().createArchiveInputStream(archive, cis)) {
							// Compressor with Archive
							return findInArchiveStream(fragment, ais);
						}
					}
					else { // Compressor only
						return StreamUtils.copyToByteArray(cis);
					}
				}
			}
			else if (StringUtils.hasText(archive)) { // Archive only
				try (ArchiveInputStream ais = new ArchiveStreamFactory().createArchiveInputStream(archive, bi)) {
					return findInArchiveStream(fragment, ais);
				}
			}
			else {
				// No compressor nor Archive
				return StreamUtils.copyToByteArray(bi);
			}

		}
	}

	/**
	 * Traverses the Archive to find either an entry that matches the modelFileNameInArchive name (if not empty) or
	 * and entry that ends in .pb if the modelFileNameInArchive is empty.
	 *
	 * @param modelFileNameInArchive Optional name of the archive entry that represents the frozen model file. If empty
	 *                               the archive will be searched for the first entry that ends in .pb
	 * @param archive Archive stream to be traversed
	 * @return
	 * @throws IOException
	 */
	private byte[] findInArchiveStream(String modelFileNameInArchive, ArchiveInputStream archive) throws IOException {
		ArchiveEntry entry;
		while ((entry = archive.getNextEntry()) != null) {
			System.out.println(entry.getName() + " : " + entry.isDirectory());

			if (archive.canReadEntryData(entry) && !entry.isDirectory()) {
				if ((StringUtils.hasText(modelFileNameInArchive) && entry.getName().endsWith(modelFileNameInArchive)) ||
						(!StringUtils.hasText(modelFileNameInArchive) && entry.getName().endsWith(this.frozenGraphFileExtension))) {
					return StreamUtils.copyToByteArray(archive);
				}
			}
		}
		throw new IllegalArgumentException("No model is found in the archive");
	}

	/**
	 * Detect the Archive and the Compressor from the file extension
	 *
	 * @param fileName File name with extension
	 * @return Returns a tuple of the detected (Archive, Compressor). Null stands for not available archive or detector.
	 * The (null, null) response stands for no Archive or Compressor discovered.
	 */
	private String[] detectArchiveAndCompressor(String fileName) {

		String normalizedFileName = fileName.trim().toLowerCase();

		if (normalizedFileName.endsWith(".tar.gz")
				|| normalizedFileName.endsWith(".tgz")
				|| normalizedFileName.endsWith(".taz")) {
			return new String[] { ArchiveStreamFactory.TAR, CompressorStreamFactory.GZIP };
		}
		else if (normalizedFileName.endsWith(".tar.bz2")
				|| normalizedFileName.endsWith(".tbz2")
				|| normalizedFileName.endsWith(".tbz")) {
			return new String[] { ArchiveStreamFactory.TAR, CompressorStreamFactory.BZIP2 };
		}
		else if (normalizedFileName.endsWith(".cpgz")) {
			return new String[] { ArchiveStreamFactory.CPIO, CompressorStreamFactory.GZIP };
		}
		else if (hasArchive(normalizedFileName)) {
			return new String[] { findArchive(normalizedFileName).get(), null };
		}
		else if (hasCompressor(normalizedFileName)) {
			return new String[] { null, findCompressor(normalizedFileName).get() };
		}
		else if (normalizedFileName.endsWith(".gzip")) {
			return new String[] { null, CompressorStreamFactory.GZIP };
		}
		else if (normalizedFileName.endsWith(".bz2")
				|| normalizedFileName.endsWith(".bz")) {
			return new String[] { null, CompressorStreamFactory.BZIP2 };
		}

		// No archived/compressed
		return new String[] { null, null };
	}

	private boolean hasArchive(String normalizedFileName) {
		return findArchive(normalizedFileName).isPresent();
	}

	private Optional<String> findArchive(String normalizedFileName) {
		return new ArchiveStreamFactory().getInputStreamArchiveNames()
				.stream().filter(arch -> normalizedFileName.endsWith("." + arch)).findFirst();
	}

	private boolean hasCompressor(String normalizedFileName) {
		return findCompressor(normalizedFileName).isPresent();
	}

	private Optional<String> findCompressor(String normalizedFileName) {
		return new CompressorStreamFactory().getInputStreamCompressorNames()
				.stream().filter(compressor -> normalizedFileName.endsWith("." + compressor)).findFirst();
	}

	public static void main(String[] args) throws Exception {
		String modelUri = "http://download.tensorflow.org/models/deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz#frozen_inference_graph.pb";
		//String modelUri = "file:/Users/ctzolov/Downloads/deeplabv3_mnv2_pascal_train_aug.zip#frozen_inference_graph.pb";
		//String modelUri = "file:/Users/ctzolov/Downloads/deeplabv3_mnv2_pascal_train_aug.zip";
		//String modelUri = "http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz";
		Resource resource = new DefaultResourceLoader().getResource(modelUri);

		//byte[] model = getFrozenModel(resource);
		//System.out.println(model.length);

		byte[] model2 = new ModelExtractor().getModel(resource);
		System.out.println(model2.length / (1024 * 1024) + "MB");

	}
}

