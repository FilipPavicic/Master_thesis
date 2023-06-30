package fer.fpavicic.jmbagDetector.dataset;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Dataset {
    private static Dataset instance;
    private INDArray images;
    private INDArray labels;

    private Dataset() {}

    public static Dataset getInstance(INDArray images, INDArray labels) {
        if (instance == null) {
            instance = new Dataset();
        }
        instance.images = images;
        instance.labels = labels;
        return instance;
    }

    public INDArray getImages() {
        return images;
    }

    public void setImages(INDArray images) {
        this.images = images;
    }

    public INDArray getLabels() {
        return labels;
    }

    public void setLabels(INDArray labels) {
        this.labels = labels;
    }

    public boolean isDefined() {
        return images != null && labels != null;
    }
}
