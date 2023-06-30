package fer.fpavicic.jmbagDetector.loaders;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import fer.fpavicic.jmbagDetector.dataset.Dataset;
import fer.fpavicic.jmbagDetector.tools.Tools;
import fer.fpavicic.jmbagDetector.visitors.DatasetVisitor;


/**
 * A batch loader for datasets, providing functionality to load images and labels in batches.
 */
public class DatasetBatchLoader{
	private List<String[]> data;
	private INDArray images; 
	private INDArray labels;
	private long imageHeight;
	private long imageWidth;
	private long imageChannels;
	int batchSize;
	boolean lazyLoading;
	ImageLoader imageLoader;

	/**
     * Constructs a DatasetBatchLoader with the specified parameters.
     *
     * @param data            the dataset as a list of string arrays
     * @param imageHeight     the height of the images in the dataset
     * @param imageWidth      the width of the images in the dataset
     * @param imageChannels   the number of channels in the images in the dataset
     * @param batchSize       the size of each batch
     * @param lazyLoading     a flag indicating whether lazy loading is enabled
     */
	public DatasetBatchLoader(List<String[]> data, long imageHeight, long imageWidth, long imageChannels,int batchSize, boolean lazyLoading ) {
		this.data = data;
		this.imageHeight = imageHeight;
		this.imageWidth = imageWidth;
		this.imageChannels = imageChannels;
		this.batchSize = batchSize;
		this.imageLoader = new ImageLoader(imageHeight, imageWidth, imageChannels);
		this.lazyLoading = lazyLoading;
		if (!this.lazyLoading) {
			images = loadImages(data, null, null);
			labels = loadLabels(data, null, null);
		}
	}

	/**
     * Loads the labels for a subset of data.
     *
     * @param subset      the subset of data as a list of string arrays
     * @param startIndex  the starting index of the subset
     * @param endIndex    the ending index of the subset
     * @return the loaded labels as an INDArray
     */
	private INDArray loadLabels(List<String[]> subset, Integer startIndex, Integer endIndex) {
		int numExamples = subset.size();
		if (startIndex == null) {
			startIndex = 0;
		}
		if (endIndex == null || endIndex > numExamples) {
			endIndex = numExamples;
		}

		int numLabels = endIndex - startIndex;
		String[] firstItem = subset.get(startIndex);
		String firstAnnotation = firstItem[0];

		INDArray firstLabel = Tools.convertStrToAnnotation(firstAnnotation);
		long labelHeight = firstLabel.size(0);
		long labelWidth = firstLabel.size(1);

		INDArray labels = Nd4j.zeros(numLabels, labelHeight, labelWidth);
		labels.put(new INDArrayIndex[]{NDArrayIndex.interval(0, numLabels), NDArrayIndex.all(), NDArrayIndex.all()}, firstLabel);

		for (int i = startIndex + 1; i < endIndex; i++) {
			String[] item = subset.get(i);
			String annotation = item[0];
			INDArray label = Tools.convertStrToAnnotation(annotation);
			labels.put(new INDArrayIndex[]{NDArrayIndex.point(i - startIndex), NDArrayIndex.all(), NDArrayIndex.all()}, label);
		}

		return labels;
	}

	/**
     * Loads the images for a subset of data.
     *
     * @param subset          the subset of data as a list of string arrays
     * @param startIndex      the starting index of the subset
     * @param endIndex        the ending index of the subset
     * @return the loaded images as an INDArray
     */
	private INDArray loadImages(List<String[]> subset, Integer startIndex, Integer endIndex) {
		int numExamples = subset.size();
		if (startIndex == null) {
			startIndex = 0;
		}
		if (endIndex == null || endIndex > numExamples) {
			endIndex = numExamples;
		}

		int numImages = endIndex - startIndex;

		INDArray images = Nd4j.zeros(numImages, imageHeight, imageWidth, imageChannels);

		for (int i = startIndex; i < endIndex; i++) {
			String[] item = subset.get(i);
			String path = item[1];
			INDArray image = null;
			try {
				image = imageLoader.loadImage(path);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			images.put(new INDArrayIndex[]{NDArrayIndex.point(i - startIndex), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()}, image);
		}
		return images;
	}
	
	/**
     * Returns the size of the dataset.
     *
     * @return the size of the dataset
     */
	public int size() {
		return data.size();
	}
	
	/**
     * Returns the number of batches in the dataset.
     *
     * @return the number of batches in the dataset
     */
	public int bathces() {
		return (int) Math.ceil(data.size() * 1.0  / this.batchSize);
	}
	
	/**
     * Returns the loaded images.
     *
     * @return the loaded images as an INDArray
     */
	public INDArray getImages() {
		if (lazyLoading) {
			INDArray batchLabels = loadLabels(data, null, null);
			return batchLabels;
		}
		return images;
	}

	/**
     * Returns the loaded labels.
     *
     * @return the loaded labels as an INDArray
     */
	public INDArray getLabels() {
		if (lazyLoading) {
			INDArray batchImages = loadImages(data, null, null);
			return batchImages;
		}
		return labels;
	}

	/**
     * Retrieves a dataset batch at the specified batch index.
     *
     * @param batchIndex  the index of the batch
     * @return a Dataset object containing the images and labels for the batch
     */
	public Dataset get(int batchIndex) {
		if (batchIndex >= this.bathces()) throw new IndexOutOfBoundsException("");
		int batchStartIndex = batchIndex * this.batchSize;
		int batchEndIndex = Math.min(batchStartIndex + this.batchSize, size());
		if (lazyLoading) {
			INDArray batchImages = loadImages(data, batchStartIndex, batchEndIndex);
			INDArray batchLabels = loadLabels(data, batchEndIndex, batchEndIndex);
			return Dataset.getInstance(batchImages, batchLabels);
		}
		INDArray batchImages = images.get(NDArrayIndex.interval(batchStartIndex, batchEndIndex), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
		INDArray batchLabels = labels.get(NDArrayIndex.interval(batchStartIndex, batchEndIndex), NDArrayIndex.all(), NDArrayIndex.all());
		return Dataset.getInstance(batchImages, batchLabels);

	}
	
	/**
     * Returns an IImageBatchLoader that can be used to load images from the dataset in batches.
     *
     * @return an IImageBatchLoader object
     */
	public IImageBatchLoader imageLoader() {
		return new IImageBatchLoader() {
			
			@Override
			public int size() {
				return DatasetBatchLoader.this.size();
			}
			
			@Override
			public INDArray get(int batchIndex) {
				return DatasetBatchLoader.this.get(batchIndex).getImages();
			}

			@Override
			public int batches() {
				// TODO Auto-generated method stub
				return DatasetBatchLoader.this.bathces();
			}
		};
	}

	/**
     * A sample main method to demonstrate the usage of the DatasetBatchLoader.
     *
     * @param args the command line arguments
     */
	public static void main(String[] args) {
		long startTime = System.currentTimeMillis();

		String labelfile = "dataset-info_corrected.csv";
		String dataPath = "D:\\FER\\IstrazivackiSeminar\\data";
		DatasetBatchLoader loader = null;
		
		

		try {
			DatasetVisitor fileVisitor = new DatasetVisitor(labelfile);
			Files.walkFileTree(Paths.get(dataPath), fileVisitor);
	        List<String[]> data = fileVisitor.getData();
			loader = new DatasetBatchLoader(data, 128, 128, 1, 32, false);
			long endTime = System.currentTimeMillis();
			long totalTime = endTime - startTime;

			INDArray images = loader.getImages();
			long totalElementsImages = images.length();
			long memoryConsumptionImages = totalElementsImages * images.data().getElementSize();

			INDArray labels = loader.getLabels();
			long totalElementsLabels = labels.length();
			long memoryConsumptionLabels = totalElementsLabels * labels.data().getElementSize();

			System.out.println("Images shape: " + Arrays.toString(images.shape()));
			System.out.println("Labels shape: " + Arrays.toString(labels.shape()));
			System.out.println("Images memory consumption: " + memoryConsumptionImages / 1024.0 / 1024 + " MB");
			System.out.println("Labels memory consumption: " + memoryConsumptionLabels / 1024.0 / 1024 + " MB");
			System.out.println("Time taken to load data: " + totalTime + " milliseconds");
		} catch (IOException e) {
			System.err.println("Failed to create Dataset: " + e.getMessage());
			e.printStackTrace();
			return;
		}


	}

}
