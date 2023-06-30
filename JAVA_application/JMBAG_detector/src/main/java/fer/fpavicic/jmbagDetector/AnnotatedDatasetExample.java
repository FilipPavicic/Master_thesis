package fer.fpavicic.jmbagDetector;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import fer.fpavicic.jmbagDetector.loaders.DatasetBatchLoader;
import fer.fpavicic.jmbagDetector.model.Model;
import fer.fpavicic.jmbagDetector.tools.Tools;
import fer.fpavicic.jmbagDetector.visitors.DatasetVisitor;

public class AnnotatedDatasetExample {
	
	public static void main(String[] args) {

		String modelPath = "D:\\FER\\diplomski_rad\\code\\test_model_mid.h5";
		String labelfile = "dataset-info_corrected.csv";
		String dataPath = "D:\\FER\\diplomski_rad\\data";
		
		main(modelPath, labelfile, dataPath);
	}

	public static void main(String modelPath, String labelfile, String dataPath) {
		
		
		Model model = new Model(modelPath, true);
		
		DatasetVisitor fileVisitor = new DatasetVisitor(labelfile);
		try {
			Files.walkFileTree(Paths.get(dataPath), fileVisitor);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        List<String[]> data = fileVisitor.getData();
        
        int imageHeight = 128;
        int imageWidth = 128;
        int imageChannels = 1;
        int batchSize = 64;
        boolean lazyLoading = false;
        
        DatasetBatchLoader datasetLoader = new DatasetBatchLoader(data, imageHeight, imageWidth, imageChannels, batchSize, lazyLoading);
        
        System.out.println("Dataset contains: " + datasetLoader.size()+" examples, divided into " + datasetLoader.bathces()+ " batches");

        INDArray concatenatedOutput = model.predict(datasetLoader.imageLoader());

		System.out.println("concatenatedOutput shape: " + Arrays.toString(concatenatedOutput.shape()));


		String[] paths = data.stream().map(s -> s[1].toString()).toArray(String[]::new);
		Tools.predictionSummary(paths, datasetLoader.getLabels(), concatenatedOutput);
		
		System.out.println("Example of one matrix");
		Tools.printRoundedOutput(concatenatedOutput.get(NDArrayIndex.point(0)));
	}


}
