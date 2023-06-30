package fer.fpavicic.jmbagDetector.metrics;
import java.util.Arrays;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.ops.NDBase;

public class AccuracyCalculator {
    public static INDArray absoluteAccuracy(INDArray yTrue, INDArray yPred, double roundValue) {
        INDArray roundedYPred = Nd4j.where(yPred.gt(roundValue), Nd4j.zerosLike(yPred), Nd4j.onesLike(yPred))[0];
        INDArray equal = yTrue.eq(roundedYPred);
        NDBase base = new NDBase();
        return base.all(equal, new int[]{1,2});
    }

    public static void main(String[] args) {
        // Example usage
        INDArray yTrue = Nd4j.createFromArray(
                new float[][][] {
                        {
                                {1, 0}, {0, 1},
                        },
                        {
                                {1, 0}, {0, 1},
                        }
                }
        );

        INDArray yPred = Nd4j.createFromArray(
                new float[][][] {
                        {
                                {0.8f, 0.2f}, {0.3f, 0.9f}
                        },
                        {
                                {0.9f, 0.9f}, {0.2f, 0.8f}
                        }
                }
        );

        double roundValue = 0.5;
        System.out.println(Arrays.toString(yTrue.shape()));
        INDArray accuracy = absoluteAccuracy(yTrue, yPred, roundValue);
        INDArray floatArray = accuracy.castTo(Nd4j.defaultFloatingPointType());

        // Calculate the mean of the float array
        float mean = floatArray.meanNumber().floatValue();

        System.out.println("Mean: " + mean);
        
        System.out.println("Accuracy: " + accuracy);
    }
}
