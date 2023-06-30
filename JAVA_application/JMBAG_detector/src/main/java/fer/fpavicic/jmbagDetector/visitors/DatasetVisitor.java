package fer.fpavicic.jmbagDetector.visitors;
import java.io.IOException;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.List;

/**
 * The DatasetVisitor class is a file visitor that traverses a directory tree and extracts data from a target CSV file.
 */

public class DatasetVisitor extends SimpleFileVisitor<Path> {
    private String targetCsvFile;
    private List<String[]> data;

    /**
     * Constructs a DatasetVisitor with the specified target CSV file.
     *
     * @param targetCsvFile the name of the target CSV file to extract data from
     */
    public DatasetVisitor(String targetCsvFile) {
        this.targetCsvFile = targetCsvFile;
        this.data = new ArrayList<>();
    }

    /**
     * Returns the extracted data from the target CSV file.
     *
     * @return the extracted data as a list of string arrays
     */
    public List<String[]> getData() {
        return data;
    }

    @Override
    public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
        if (file.getFileName().toString().equals(targetCsvFile)) {
            List<String> lines = Files.readAllLines(file);
            for (String line : lines) {
                String[] values = line.split(",");
                if (values.length >= 2) {
                    String annotation = values[1].trim();
                    String path = values[0].trim();
                    String fullPath = file.getParent().toString();
                    String prefixedValue = fullPath + "\\" + path;
                    String[] item = {annotation, prefixedValue};
                    data.add(item);
                }
            }
        }
        return FileVisitResult.CONTINUE;
    }

    @Override
    public FileVisitResult visitFileFailed(Path file, IOException exc) {
        System.err.println("Failed to visit file: " + file);
        return FileVisitResult.CONTINUE;
    }

    @Override
    public FileVisitResult preVisitDirectory(Path dir, BasicFileAttributes attrs) throws IOException {
        return FileVisitResult.CONTINUE;
    }

    /**
     * A main method for testing the DatasetVisitor class.
     *
     * @param args command-line arguments (not used)
     */
    public static void main(String[] args) {
        String targetCsvFile = "dataset-info_corrected.csv";
        String startingDir = "D:\\FER\\IstrazivackiSeminar\\data";
        
        DatasetVisitor fileVisitor = new DatasetVisitor(targetCsvFile);
        try {
            Files.walkFileTree(Paths.get(startingDir), fileVisitor);
            List<String[]> data = fileVisitor.getData();
            System.out.println("Number of loaded data: " + data.size() );
            System.out.println("10 examples:");
            int i = 0;
            for (String[] item : data) {
                String annotation = item[0];
                String path = item[1];
                System.out.println("annotation: " + annotation + ", path: " + path);
                i++;
                if (i >= 10) break;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
