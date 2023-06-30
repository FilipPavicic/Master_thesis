package fer.fpavicic.jmbagDetector;

import java.io.IOException;

import org.nd4j.linalg.api.ndarray.INDArray;

import fer.fpavicic.jmbagDetector.loaders.ImageBatchLoader;
import fer.fpavicic.jmbagDetector.loaders.ImageLoader;
import fer.fpavicic.jmbagDetector.model.Model;
import fer.fpavicic.jmbagDetector.tools.Tools;

public class NonAnnotatedDatasetExample {
	public static void main(String[] args) {
		String modelPath = "D:\\FER\\diplomski_rad\\code\\test_model_big.h5";
		String fileExtension = "png";
		String dataPath = "D:\\FER\\diplomski_rad\\data";
		
		main(modelPath, fileExtension, dataPath);
	}
	public static void main(String modelPath, String fileExtension, String dataPath) {	
		Model model = new Model(modelPath, true);
		String[] imagesPaths = Tools.loadFilePaths(dataPath, fileExtension);
		System.out.println("Loaded " + imagesPaths.length  + " images");
        
        int imageHeight = 128;
        int imageWidth = 128;
        int imageChannels = 1;
        int batchSize = 32;
        
        ImageLoader loader = new ImageLoader(imageHeight, imageWidth, imageChannels);
        INDArray images = null;
		try {
			images = loader.loadImages(imagesPaths);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        INDArray concatenatedOutput = model.predict(new ImageBatchLoader(images, batchSize));
		Tools.outputPrediciton("resources/prediction.csv", imagesPaths, concatenatedOutput, true, true);
	}
}
