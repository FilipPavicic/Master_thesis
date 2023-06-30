package fer.fpavicic.jmbagDetector.model;

import java.io.IOException;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import fer.fpavicic.jmbagDetector.loaders.IImageBatchLoader;

/**
 * Represents a deep learning model used for predictions.
 */
public class Model {

    String modelPath;
    ComputationGraph model = null;
    boolean debugFlag;

    /**
     * Constructs a new Model object with the specified model path and debug flag.
     *
     * @param modelPath  the path to the model file
     * @param debugFlag  a flag indicating whether debug information should be printed
     */
    public Model(String modelPath, boolean debugFlag) {
        this.modelPath = modelPath;
        this.debugFlag = debugFlag;
        try {
            model = KerasModelImport.importKerasModelAndWeights(modelPath);
        } catch (IOException | UnsupportedKerasConfigurationException | InvalidKerasConfigurationException e) {
        	System.err.println("Error occurred while importing the model: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
            
        }
    }

    /**
     * Performs predictions on the given image batch loader.
     *
     * @param loader  the image batch loader
     * @return the predicted output as an INDArray
     */
    public INDArray predict(IImageBatchLoader loader) {
        long startTime = System.currentTimeMillis();
        INDArray concatenatedOutput = null;

        long counter = 0;
        
        if (debugFlag) {
            System.out.println("Starting prediction with " + loader.size() + " images.");
        }

        for (int i = 0; i < loader.batches(); i++) {
            
            INDArray images = loader.get(i);
            long startLapTime = System.currentTimeMillis();
            INDArray output = model.outputSingle(images);
            long endTime = System.currentTimeMillis();

            if (concatenatedOutput == null) {
                concatenatedOutput = output;
            } else {
                concatenatedOutput = Nd4j.concat(0, concatenatedOutput, output);
            }
            counter += output.shape()[0];
            
            long totalTime = endTime - startTime;
            long totalLapTime = endTime - startLapTime;
            if (debugFlag) {
                System.out.println("Processed " + (i+1) + "/" + loader.batches() + " batches" +
                        " (" + counter + "/" + loader.size() + "), " +
                        "Total pass time: " + totalTime + "ms" +
                        " (lap time: " + totalLapTime + "ms)");
            }
        }
        
        long endTime = System.currentTimeMillis();
        long totalTime = endTime - startTime;

        if (debugFlag) {
            System.out.println("Prediction completed in " + totalTime + "ms.");
        }
        concatenatedOutput = concatenatedOutput.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(concatenatedOutput.size(3) - 1));
        return concatenatedOutput;
    }

}
