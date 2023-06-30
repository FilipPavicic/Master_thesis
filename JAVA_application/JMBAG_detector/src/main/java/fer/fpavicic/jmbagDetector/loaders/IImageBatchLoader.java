package fer.fpavicic.jmbagDetector.loaders;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface IImageBatchLoader {
	
	/**
	 * Returns the loaded images.
	 *
	 * @return the loaded images as an INDArray
	 */
	public int size();
	
	/**
	 * Returns the size of the image dataset.
	 *
	 * @return the size of the image dataset
	 */
	public int batches();
	
	/**
	 * Returns the number of batches in the image dataset.
	 *
	 * @return the number of batches in the image dataset
	 */
	public INDArray get(int batchIndex);

}
