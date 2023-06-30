import tkinter as tk
from PIL import Image, ImageTk
import os
import json
import pickle
import numpy as np
from pathlib import Path
import AnnotationData


def load_image(file_path, image_label):
    image = Image.open(folder / file_path)
    image = ImageTk.PhotoImage(image)
    image_label.config(image=image)
    image_label.image = image


def show_next_image(event=None):
    global current_image_index
    current_image_index += 1
    if current_image_index >= len(annotations):
        current_image_index = 0
    current_image, current_annotaiton = annotations[current_image_index]
    load_image(current_image, image_label)
    update_squares(current_annotaiton)
    image_name_label.config(text=current_image)


def show_prev_image(event=None):
    global current_image_index
    current_image_index -= 1
    if current_image_index < 0:
        current_image_index = len(annotations) - 1
    current_image, current_annotaiton = annotations[current_image_index]
    load_image(current_image, image_label)
    update_squares(current_annotaiton)
    image_name_label.config(text=current_image)


def update_squares(annotation):
    for i, row in enumerate(squares):
        for j, (rectangle, square) in enumerate(row):
            square.itemconfig(rectangle, fill='white')
            if annotation[i][j] == 1:
                square.itemconfig(rectangle, fill='green')


# Read annotations
folder_str = "data/studentidmatrix-dataset03/dataset03"
folder = Path(folder_str)
annotations = AnnotationData.AnnotationData(folder_str + "/dataset-info.pkl")
# Get list of images
current_image_index = 0
current_image, current_annotaiton = annotations[current_image_index]

# Create GUI
root = tk.Tk()
root.title("Annotation Checker")
root.bind("<Right>", show_next_image)
root.bind("<Left>", show_prev_image)

# Add image label
image_label = tk.Label(root)
image_label.grid(row=0, column=0, rowspan=10, padx=10, pady=10)
load_image(current_image, image_label)

# Adding the squares
squares = []
for i in range(10):
    row = []
    for j in range(10):
        square = tk.Canvas(root, height=30, width=30, highlightthickness=0)
        rectagle = square.create_rectangle(0, 0, 30, 30, fill='white', outline='black')
        square.create_text(15, 15, text=str(i))
        square.grid(row=i, column=j+1, padx=2, pady=2)
        square.bind("<Button-1>", lambda event, i=i, j=j: on_square_click(event, i, j))
        row.append((rectagle, square))
    squares.append(row)

update_squares(current_annotaiton)
# Adding the OK button


def on_ok():
    table = []
    for i, row in enumerate(squares):
        row_values = []
        for j, square in enumerate(row):
            row_values.append(square.cget("background") == "green")
        table.append(row_values)
    print(table)


image_name_label = tk.Label(root, text=current_image)
image_name_label.grid(row=11, column=0, columnspan=11, pady=10)

# Function to handle square clicks


def on_square_click(event, i, j):
    rectagle, square = squares[i][j]

    if square.itemcget(rectagle, "fill") == "white":
        square.itemconfig(rectagle, fill='green')
        value = 1
    else:
        square.itemconfig(rectagle, fill='white')
        value = 0
    annotations.update_annotations(current_image_index, i, j, value)


# Running the GUI
root.mainloop()
