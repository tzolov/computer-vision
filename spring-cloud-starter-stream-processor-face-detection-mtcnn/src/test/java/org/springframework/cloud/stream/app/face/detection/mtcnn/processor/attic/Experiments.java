package org.springframework.cloud.stream.app.face.detection.mtcnn.processor.attic;

import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import java.nio.FloatBuffer;
import java.util.concurrent.TimeUnit;

import static org.bytedeco.javacpp.tensorflow.*;


/**
 * @author Christian Tzolov
 */
public class Experiments {

	public static void main(String[] args) {
		Scope root = Scope.NewRootScope();

		Tensor condition = Tensor.create(new float[] { 0, 1, 1, 0, 0, 0, 1, 0, 0 }, new TensorShape(3, 3));

		Where where = new Where(root, new Input(condition));

		GraphDef def = new GraphDef();
		Status s = root.ToGraphDef(def);
		if (!s.ok()) {
			throw new RuntimeException(s.error_message().getString());
		}

	}

	static class Options {
		int num_concurrent_sessions = 10; // The number of concurrent sessions
		int num_concurrent_steps = 1;    // The number of concurrent steps
		int num_iterations = 100;         // Each step repeats this many times
		boolean use_gpu = false;          // Whether to use gpu in the training
	}

	static void ConcurrentSteps(final Options opts, GraphDef def) throws Exception {
		// Creates a session.
		SessionOptions options = new SessionOptions();
		final Session session = new Session(options);

		if (options.target() == null) {
			SetDefaultDevice(opts.use_gpu ? "/gpu:0" : "/cpu:0", def);
		}

		Status s = session.Create(def);
		if (!s.ok()) {
			throw new Exception(s.error_message().getString());
		}

		// Spawn M threads for M concurrent steps.
		int M = opts.num_concurrent_steps;
		ExecutorService step_threads = Executors.newFixedThreadPool(M);

		for (int step = 0; step < M; step++) {
			final int m = step;
			step_threads.submit((Callable<Void>) () -> {
				// Randomly initialize the input.
				Tensor x = new Tensor(DT_FLOAT, new TensorShape(2, 1));
				FloatBuffer x_flat = x.createBuffer();
				x_flat.put(0, (float) Math.random());
				x_flat.put(1, (float) Math.random());
				float inv_norm = 1 / (float) Math.sqrt(x_flat.get(0) * x_flat.get(0) + x_flat.get(1) * x_flat.get(1));
				x_flat.put(0, x_flat.get(0) * inv_norm);
				x_flat.put(1, x_flat.get(1) * inv_norm);

				// Iterations.
				TensorVector outputs = new TensorVector();
				for (int iter = 0; iter < opts.num_iterations; iter++) {
					outputs.resize(0);
					Status s1 = session.Run(new StringTensorPairVector(new String[] { "x" }, new Tensor[] { x }),
							new StringVector("y:0", "y_normalized:0"), new StringVector(), outputs);
					if (!s1.ok()) {
						throw new Exception(s1.error_message().getString());
					}
					assert outputs.size() == 2;

					Tensor y = outputs.get(0);
					Tensor y_norm = outputs.get(1);
					// Print out lambda, x, and y.
					//System.out.printf("%06d %s\n", m, DebugString(x, y));
					// Copies y_normalized to x.
					x.put(y_norm);
				}
				return null;
			});
		}

		step_threads.shutdown();
		step_threads.awaitTermination(1, TimeUnit.MINUTES);
		s = session.Close();
		if (!s.ok()) {
			throw new Exception(s.error_message().getString());
		}
	}

}
