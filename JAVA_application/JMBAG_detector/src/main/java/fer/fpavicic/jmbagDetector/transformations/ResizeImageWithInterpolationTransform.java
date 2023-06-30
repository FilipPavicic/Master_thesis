package fer.fpavicic.jmbagDetector.transformations;


import static org.bytedeco.opencv.global.opencv_imgproc.resize;

import java.util.Random;

import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import org.datavec.image.data.ImageWritable;
import org.datavec.image.transform.BaseImageTransform;

public class ResizeImageWithInterpolationTransform extends BaseImageTransform<Mat> {

    private int newHeight;
    private int newWidth;
    private int interpolation;

    private int srch;
    private int srcw;

    /**
     * Returns new ResizeImageTransform object
     *
     * @param newWidth new Width for the outcome images
     * @param newHeight new Height for outcome images
     */
    public ResizeImageWithInterpolationTransform(int newWidth, int newHeight, int interpolation) {
    	super(null);

        this.newWidth = newWidth;
        this.newHeight = newHeight;
        this.interpolation = interpolation;
        this.converter = new OpenCVFrameConverter.ToMat();
    }


    /**
     * Takes an image and returns a transformed image.
     * Uses the random object in the case of random transformations.
     *
     * @param image  to transform, null == end of stream
     * @param random object to use (or null for deterministic)
     * @return transformed image
     */
    @Override
    protected ImageWritable doTransform(ImageWritable image, Random random) {
        if (image == null) {
            return null;
        }
        Mat mat = converter.convert(image.getFrame());
        Mat result = new Mat();
        srch = mat.rows();
        srcw = mat.cols();
        resize(mat, result, new Size(newWidth, newHeight), 0, 0, interpolation);
        return new ImageWritable(converter.convert(result));
    }

    @Override
    public float[] query(float... coordinates) {
        float[] transformed = new float[coordinates.length];
        for (int i = 0; i < coordinates.length; i += 2) {
            transformed[i    ] = newWidth * coordinates[i    ] / srcw;
            transformed[i + 1] = newHeight * coordinates[i + 1] / srch;
        }
        return transformed;
    }
}
