import pickle
import numpy as np
from pathlib import Path
import glob
import csv


class AnnotationData:
    def __init__(self, annotationfile):
        self.annotations = []
        self.annotationFile = annotationfile
        self.load_annotations(Path(annotationfile))

    def load_annotations(self, file_path: Path):
        assert file_path.suffix == ".pkl"
        with open(file_path, "rb") as file:
            self.annotations = pickle.load(file)

    @staticmethod
    def convertAnotation(input_file, output_file):
        annotations = []
        with open(input_file, "r") as file:
            for line in file:
                name, annotationStr = line.strip().split("\t")
                annotation = np.zeros((10, 10), dtype=np.int8)
                try:
                    indexes_x = np.arange(0, 10)
                    indexes_y = np.array([int(i) for i in annotationStr])
                except Exception as e:
                    print(f"file {file} ended with exception: {e}")
                annotation[indexes_y, indexes_x] = 1
                annotations.append((name, annotation))

        with open(output_file, "wb") as file:
            pickle.dump(annotations, file)

    def update_annotations(self, index, i, j, value):

        self.annotations[index][1][i, j] = value
        self.save_annotations()

    def save_annotations(self):
        with open(self.annotationFile, "wb") as file:
            pickle.dump(self.annotations, file)

    def save_annotations_csv(self, output_path):
        data = [(name, AnnotationData.convertAnnotationToStr(annt)) for name, annt in self.annotations]
        with open(output_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(data)

    def __getitem__(self, item):
        return self.annotations[item]

    def __len__(self):
        return len(self.annotations)

    @staticmethod
    def convertAnnotationToStr(annotation: np.ndarray):
        if len(annotation.shape) == 3:
            annotation = annotation[..., -1]
        result = ''
        for i in range(annotation.shape[1]):
            column = annotation[:, i]
            ones = np.argwhere(column == 1)
            if ones.size == 0:
                result += 'X'
            elif ones.size == 1:
                result += str(np.argmax(column))
            else:
                result += '[' + ''.join(str(x[0]) for x in ones) + ']'
        return result

    # @staticmethod
    # def convertStrToAnnotation(annotation_str : str, cols = 10):
    #     annotation = np.zeros((10, cols))
    #     index = 0
    #     in_bracket = False
    #     for ch in annotation_str:
    #         if ch == "X":
    #             assert not in_bracket, "X can not be in brackets"
    #             pass
    #         elif ch == "[":
    #             assert not in_bracket, "[ can not be in brackets"
    #             in_bracket = True
    #         elif ch == "[":
    #             assert in_bracket, "bracket is not open"
    #             in_bracket = False
    #         elif


# if __name__ == "__main__":
    # input_file = Path("studentidmatrix-dataset09/dataset09/dataset-info.txt")
    # output_file = input_file.parent / (input_file.stem + ".pkl")
    # AnnotationData.convertAnotation(input_file, output_file)
