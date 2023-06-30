package fer.fpavicic.jmbagDetector.loaders;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class ImageBatchLoader implements IImageBatchLoader{
	private INDArray images;
	int batchSize;

	public ImageBatchLoader(INDArray images, int batchSize) {
		this.images = images;
		this.batchSize = batchSize;
	}

	/**
	 * Returns the loaded images.
	 *
	 * @return the loaded images as an INDArray
	 */
	public INDArray getImages() {
		return images;
	}

	@Override
	public int size() {
		return (int) images.shape()[0];
	}

	@Override
	public int batches() {
		return (int) Math.ceil(images.shape()[0] * 1.0  / this.batchSize);
	}

	@Override
	public INDArray get(int batchIndex) {
		if (batchIndex >= this.size()) throw new IndexOutOfBoundsException("");
		int batchStartIndex = batchIndex * this.batchSize;
		int batchEndIndex = (int) Math.min(batchStartIndex + this.batchSize, images.shape()[0]);
		INDArray batchImages = images.get(NDArrayIndex.interval(batchStartIndex, batchEndIndex), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
		return batchImages;

	}
}
