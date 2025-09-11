package org.example;

import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.translate.TranslateException;
import ai.djl.inference.Predictor;
import ai.djl.translate.Batchifier;

import java.io.IOException;
import java.nio.file.Paths;

//mvn exec:java -Dexec.mainClass="org.example.Main"

public class Main {

    public static void main(String[] args) {

        Translator<float[][], float[][]> myTranslator = new Translator<float[][], float[][]>() {

            @Override
            public NDList processInput(TranslatorContext ctx, float[][] input) throws Exception {
                NDManager manager = ctx.getNDManager();

                int height = input.length;
                int width = input[0].length;

                float[] flat = new float[height * width];
                for (int i = 0; i < height; i++) {
                    System.arraycopy(input[i], 0, flat, i * width, width);
                }

                NDArray x = manager.create(flat, new Shape(height, width));

                // Add batch and channel dims -> [1,1,36,112]
                x = x.expandDims(0).expandDims(0);

                return new NDList(x);
            }

            @Override
            public float[][] processOutput(TranslatorContext ctx, NDList list) throws Exception {
                NDArray result = list.get(0);

                // Remove batch and channel dims -> [36,112]
                result = result.squeeze(); 

                // Convert to 1D float array
                float[] flat = result.toFloatArray();

                // Reshape manually into 2D array
                long[] shape = result.getShape().getShape();
                int height = (int) shape[0];
                int width = (int) shape[1];
                float[][] output2d = new float[height][width];
                for (int i = 0; i < height; i++) {
                    System.arraycopy(flat, i * width, output2d[i], 0, width);
                }

                return output2d;
            }

            @Override
            public Batchifier getBatchifier() {
                return null; // no batching
            }
        };

        Criteria<float[][], float[][]> myModelCriteria = Criteria.builder()
                .setTypes(float[][].class, float[][].class)
                .optModelPath(Paths.get("nets/cnn_autoenc_sector1_default.pt"))
                .optEngine("PyTorch") //PyTorch
                .optTranslator(myTranslator)
                .optProgress(new ProgressBar())
                .build();

        try {
            ZooModel<float[][], float[][]> model = myModelCriteria.loadModel();
            Predictor<float[][], float[][]> predictor = model.newPredictor();

            // Dummy input: almost straight track
            float[][] dummyInput = new float[36][112];
            for (int y = 0; y < 36; y++) {
                int x = 50 + (y / 10); // slightly bending track
                dummyInput[y][x] = 1.0f;
            }

            float[][] output = predictor.predict(dummyInput);

            System.out.println("Output shape: [" + output.length + "," + output[0].length + "]");
            System.out.println("Sample output values:");
            for (int i = 0; i < 36; i++) {
                for (int j = 0; j < 112; j++) {
                    System.out.printf("%.3f ", output[i][j]);
                }
                System.out.println();
            }

        } catch (IOException | ModelNotFoundException | MalformedModelException | TranslateException e) {
            throw new RuntimeException(e);
        }
    }
}
