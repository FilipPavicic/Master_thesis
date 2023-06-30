package fer.fpavicic.jmbagDetector.loaders;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;

import org.bytedeco.javacv.Frame;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.opencv_core.Mat;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import fer.fpavicic.jmbagDetector.transformations.ResizeImageWithInterpolationTransform;

/**
 * The ImageLoader class is responsible for loading images and converting them into INDArray format.
 */
public class ImageLoader {
	NativeImageLoader nativeImageLoader;
	
	/**
     * Constructs an ImageLoader with the specified image properties and interpolation method.
     *
     * @param height        the height of the output image
     * @param width         the width of the output image
     * @param channels      the number of channels in the output image
     * @param interpolation the interpolation method for resizing the image
     */
	public ImageLoader(long height, long width, long channels, int interpolation) {
		nativeImageLoader =  new NativeImageLoader(height, width,channels, new ResizeImageWithInterpolationTransform((int) width, (int) height, interpolation));
	}
	
	/**
     * Constructs an ImageLoader with the specified image properties using default interpolation method (CV_INTER_AREA).
     *
     * @param height   the height of the output image
     * @param width    the width of the output image
     * @param channels the number of channels in the output image
     */
	public ImageLoader(long height, long width, long channels) {
		this(height, width, channels, opencv_imgproc.CV_INTER_AREA);
	}
	
	/**
     * Loads a single image from the specified file path and converts it into an INDArray.
     *
     * @param filePath the path to the image file
     * @return the loaded image as an INDArray
     * @throws IOException if an I/O error occurs
     */
	public INDArray loadImage(String f) throws IOException {
		return loadImages(f);
	}
	
	/**
     * Loads multiple images from the specified file paths and converts them into an INDArray.
     *
     * @param filePaths the paths to the image files
     * @return the loaded images as an INDArray
     * @throws IOException if an I/O error occurs
     */
	public INDArray loadImages(String... fs) throws IOException {
		INDArray images = null;
		for(var f : fs) {
			INDArray output = nativeImageLoader.asMatrix(f).div(255.0).permute(0, 2, 3, 1);
			if (images == null) {
				images = output;
		    } else {
		    	images = Nd4j.concat(0, images, output);
		    }
		}
		return images;
	}
	
	/**
     * Loads a single image from the specified input stream and converts it into an INDArray.
     *
     * @param inputStream the input stream of the image
     * @return the loaded image as an INDArray
     * @throws IOException if an I/O error occurs
     */
	public INDArray loadImage(InputStream f) throws IOException {
		return loadImages(f);
	}
	
	/**
     * Loads multiple images from the specified input streams and converts them into an INDArray.
     *
     * @param inputStreams the input streams of the images
     * @return the loaded images as an INDArray
     * @throws IOException if an I/O error occurs
     */
	public INDArray loadImages(InputStream... fs) throws IOException {
		INDArray images = null;
		for(var f : fs) {
			INDArray output = nativeImageLoader.asMatrix(f).div(255.0).permute(0, 2, 3, 1);
			if (images == null) {
				images = output;
		    } else {
		    	images = Nd4j.concat(0, images, output);
		    }
		}
		return images;
	}
	
	/**
     * Loads a single image from the specified file and converts it into an INDArray.
     *
     * @param file the image file
     * @return the loaded image as an INDArray
     * @throws IOException if an I/O error occurs
     */
	public INDArray loadImage(File f) throws IOException {
		return loadImages(f);
	}
	
	/**
     * Loads multiple images from the specified files and converts them into an INDArray.
     *
     * @param files the image files
     * @return the loaded images as an INDArray
     * @throws IOException if an I/O error occurs
     */
	public INDArray loadImages(File... fs) throws IOException {
		INDArray images = null;
		for(var f : fs) {
			INDArray output = nativeImageLoader.asMatrix(f).div(255.0).permute(0, 2, 3, 1);
			if (images == null) {
				images = output;
		    } else {
		    	images = Nd4j.concat(0, images, output);
		    }
		}
		return images;
	}
	
	/**
     * Loads a single image from the specified OpenCV Mat object and converts it into an INDArray.
     *
     * @param mat the OpenCV Mat object representing the image
     * @return the loaded image as an INDArray
     * @throws IOException if an I/O error occurs
     */
	public INDArray loadImage(Mat f) throws IOException {
		return loadImages(f);
	}
	
	/**
     * Loads multiple images from the specified OpenCV Mat objects and converts them into an INDArray.
     *
     * @param mats the OpenCV Mat objects representing the images
     * @return the loaded images as an INDArray
     * @throws IOException if an I/O error occurs
     */
	public INDArray loadImages(Mat... fs) throws IOException {
		INDArray images = null;
		for(var f : fs) {
			INDArray output = nativeImageLoader.asMatrix(f).div(255.0).permute(0, 2, 3, 1);
			if (images == null) {
				images = output;
		    } else {
		    	images = Nd4j.concat(0, images, output);
		    }
		}
		return images;
	}
	
	/**
     * Loads a single image from the specified JavaCV Frame object and converts it into an INDArray.
     *
     * @param frame the JavaCV Frame object representing the image
     * @return the loaded image as an INDArray
     * @throws IOException if an I/O error occurs
     */
	public INDArray loadImage(Frame f) throws IOException {
		return loadImages(f);
	}
	
	 /**
     * Loads multiple images from the specified JavaCV Frame objects and converts them into an INDArray.
     *
     * @param frames the JavaCV Frame objects representing the images
     * @return the loaded images as an INDArray
     * @throws IOException if an I/O error occurs
     */
	public INDArray loadImages(Frame... fs) throws IOException {
		INDArray images = null;
		for(var f : fs) {
			INDArray output = nativeImageLoader.asMatrix(f).div(255.0).permute(0, 2, 3, 1);
			if (images == null) {
				images = output;
		    } else {
		    	images = Nd4j.concat(0, images, output);
		    }
		}
		return images;
	}
	
}
