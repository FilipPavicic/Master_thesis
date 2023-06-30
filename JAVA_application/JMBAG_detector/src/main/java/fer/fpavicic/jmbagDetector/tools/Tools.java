package fer.fpavicic.jmbagDetector.tools;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import fer.fpavicic.jmbagDetector.metrics.AccuracyCalculator;

public class Tools {
	
	/**
     * Prints the rounded output values of an INDArray.
     *
     * @param output the INDArray containing the output values
     */
	public static void printRoundedOutput(INDArray output) {
		long numRows = output.size(0);
		long numCols = output.size(1);

		for (int i = 0; i < numRows; i++) {
			for (int j = 0; j < numCols; j++) {
				double value = output.getDouble(i, j);
				System.out.print(Math.round(value) + " ");
			}
			System.out.println();
		}
	}
	
	/**
     * Converts an annotation INDArray to a string representation.
     *
     * @param annotation the annotation INDArray
     * @return the string representation of the annotation
     */
	public static String convertAnnotationToStr(INDArray annotation) {

        StringBuilder result = new StringBuilder();
        long numCols = annotation.size(1);

        for (int j = 0; j < numCols; j++) {
            INDArray column = annotation.get(NDArrayIndex.all(), NDArrayIndex.point(j));
            INDArray indices = Nd4j.where(column.gt(0), null, null)[0];
            if (indices.length() == 0) {
                result.append("X");
            } else if (indices.length() == 1) {
                result.append(indices.getInt(0));
            } else {
                result.append("[");
                for (int k = 0; k < indices.length(); k++) {
                    result.append(indices.getInt(k));
                }
                result.append("]");
            }
        }

        return result.toString();
    }
	
	/**
     * Converts multiple annotation INDArrays to an array of string representations.
     *
     * @param annotations the array of annotation INDArrays
     * @param round       flag indicating whether to round the annotations
     * @return the array of string representations of the annotations
     */
	public static String[] convertMultipleAnnotationToStr(INDArray annotations, boolean round) {
		if (round) annotations = Transforms.round(annotations);
		String[] results = new String[(int) annotations.shape()[0]];
		for(int i = 0; i < annotations.shape()[0]; i++) {
			INDArray annotation = annotations.get(NDArrayIndex.point(i));
			results[i] = convertAnnotationToStr(annotation);
		}
		return results;
	}
	
	/**
     * Converts a string representation of an annotation to an INDArray.
     *
     * @param annotationStr the string representation of the annotation
     * @return the INDArray representation of the annotation
     */
	public static INDArray convertStrToAnnotation(String annotationStr) {
	    String[] characters = annotationStr.split("");
	    int numCols = 0;

	    boolean insideBrackets = false;

	    for (String character : characters) {
	        if (character.equals("[")) {
                insideBrackets = true;
                numCols++;
	        } else if (character.equals("]")) {
	            insideBrackets = false;
	        } else if (!insideBrackets) {
	            numCols++;
	        }
	    } 

	    INDArray annotation = Nd4j.zeros(10, numCols);

	    int colIndex = 0;
	    insideBrackets = false;

	    for (String character : characters) {
	        if (character.equals("[")) {
	            if (insideBrackets) {
	                throw new IllegalStateException("Opening bracket [ cannot be nested inside another bracket.");
	            }
	            insideBrackets = true;
	        } else if (character.equals("]")) {
	            if (!insideBrackets) {
	                throw new IllegalStateException("Closing bracket ] found before opening bracket [.");
	            }
	            insideBrackets = false;
	            colIndex++;
	        } else if (character.equals("X")) {
	            if (insideBrackets) {
	                throw new IllegalStateException("Invalid character X found inside brackets [].");
	            }
	            colIndex++;
	        } else {
	        	try {
	                int row = Integer.parseInt(character);
	                annotation.putScalar(row, colIndex, 1);
	                if (!insideBrackets) colIndex++;
	            } catch (NumberFormatException e) {
	                throw new IllegalArgumentException("Invalid character found: " + character);
	            }
	        }
	    }

	    return annotation;
	}
	
	/**
     * Displays a summary of the prediction results.
     *
     * @param paths   the array of file paths
     * @param yTrue   the true labels
     * @param yPred   the predicted labels
     */
	public static void predictionSummary(String[] paths, INDArray yTrue, INDArray yPred) {
		INDArray accuracy = AccuracyCalculator.absoluteAccuracy(yTrue, yPred, 0.5);
		if (paths.length != accuracy.shape()[0]) throw new IllegalArgumentException("Sizes do not match");
		
		INDArray floatArray = accuracy.castTo(Nd4j.defaultFloatingPointType());
		float accMean = floatArray.meanNumber().floatValue();

		String[] yTrueStrings = convertMultipleAnnotationToStr(yTrue, false);
		String[] yPredStrings = convertMultipleAnnotationToStr(yPred, true);
		
		System.out.println("Prediction summary:");
		System.out.println("Number of images: " + accuracy.shape()[0]);
		System.out.println("Mean accuracy: " + accMean);
		System.out.println("Prediction for each label: ");
		for (int i = 0; i < paths.length; i++) {
			System.out.println("["+ accuracy.get(NDArrayIndex.point(i))+"] Image:" + paths[i] + ", True label: " + yTrueStrings[i] + ", predicted label: " + yPredStrings[i]);
			
		}
	}
	
	/**
     * Outputs the prediction results to a CSV file and/or prints them to the console.
     *
     * @param csvFilePath the file path of the CSV file
     * @param names       the array of names
     * @param yPred       the predicted labels
     * @param save        flag indicating whether to save the results to a CSV file
     * @param print       flag indicating whether to print the results to the console
     */
	public static void outputPrediciton(String csvFilePath, String[] names, INDArray yPred, boolean save, boolean print) {
		String[] yPredStrings = convertMultipleAnnotationToStr(yPred, true);
		
		if (!save) {
			for (int i = 0; i < names.length; i++) {
				if (print) System.out.println(names[i] + "," + yPredStrings[i]);
			}
			return;
		}
		
		try (BufferedWriter writer = new BufferedWriter(new FileWriter(csvFilePath))) {
			for (int i = 0; i < names.length; i++) {
				writer.write(names[i] + "," + yPredStrings[i] + "\n");
				if (print) System.out.println(names[i] + "," + yPredStrings[i]);
			}
			

		} catch (IOException e) {
			System.err.println("Error occurred while saving the prediction results to CSV file: " + e.getMessage());
			e.printStackTrace();
		}
	}
	
	/**
	 * Loads all file paths with a specific file extension in a folder and its subfolders.
	 *
	 * @param folderPath    the path of the folder
	 * @param fileExtension the desired file extension (e.g., ".txt", ".jpg", ".csv")
	 * @return an array of file paths matching the specified file extension
	 */
	public static String[] loadFilePaths(String folderPath, String fileExtension) {
        List<String> filePaths = new ArrayList<>();
        File folder = new File(folderPath);
        if (folder.exists() && folder.isDirectory()) {
            loadFilePathsRecursive(folder, fileExtension, filePaths);
        }
        return filePaths.toArray(new String[0]);
    }

    private static void loadFilePathsRecursive(File folder, String fileExtension, List<String> filePaths) {
        File[] files = folder.listFiles();
        if (files != null) {
            for (File file : files) {
                if (file.isDirectory()) {
                    loadFilePathsRecursive(file, fileExtension, filePaths);
                } else if (file.isFile() && file.getName().toLowerCase().endsWith(fileExtension.toLowerCase())) {
                    filePaths.add(file.getAbsolutePath());
                }
            }
        }
    }
}
