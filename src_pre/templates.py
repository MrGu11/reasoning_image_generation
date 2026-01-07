"""
templates.py

每个 template 返回一个 dict:
{
  "id": str, "name": str,
  "generate": function(rng, matrix_size) -> (matrix_specs, correct_index, meta)
}
matrix_specs: list of length matrix_size*matrix_size, each cell is a list of objects
object format: {"type":"circle"/"square"/"triangle", "x":..., "y":..., "size":..., "angle":..., "color":...}
"""
import random
import math

SHAPES = ["circle", "square", "triangle"]

def simple_rotate_generate(rng, matrix_size=3):
    """
    A simple template: each row rotates a base shape by 0, 90, 180 degrees.
    Last cell missing, answer is the next rotation.
    """
    base_shape = rng.choice(SHAPES)
    size = rng.randint(20, 40)
    colors = ["black"]
    # center positions normalized (we will render ignoring absolute coords here)
    def make_obj(shape, angle, color="black"):
        return {"type": shape, "angle": angle, "size": size, "color": color}

    matrix = []
    # for 3x3, fill first 8, last is placeholder
    steps = [0, 90, 180]
    for r in range(matrix_size):
        for c in range(matrix_size):
            idx = r * matrix_size + c
            if idx == matrix_size*matrix_size - 1:
                matrix.append([])  # blank to fill
            else:
                angle = steps[c % len(steps)]
                matrix.append([make_obj(base_shape, angle)])
    # correct answer is rotation continuing the row pattern
    # For last row, last col should be steps[(matrix_size-1) % len(steps)]
    correct_angle = steps[(matrix_size-1) % len(steps)]
    answer = {"type": base_shape, "angle": correct_angle, "size": size, "color": "black"}
    meta = {"template":"simple_rotate", "base_shape": base_shape}
    return matrix, answer, meta

def count_increment_generate(rng, matrix_size=3):
    """
    Count-based template: left->right increases number of items (1,2,3) per row.
    Last cell blank, answer: correct number of shapes.
    """
    matrix = []
    size = 18
    base_shape = rng.choice(SHAPES)
    for r in range(matrix_size):
        for c in range(matrix_size):
            idx = r * matrix_size + c
            if idx == matrix_size*matrix_size - 1:
                matrix.append([])
            else:
                count = (c % matrix_size) + 1  # 1,2,3
                objs = [{"type": base_shape, "angle": 0, "size": size, "color":"black"} for _ in range(count)]
                matrix.append(objs)
    answer = [{"type": base_shape, "angle": 0, "size": size, "color":"black"} for _ in range(matrix_size)]
    meta = {"template":"count_increment", "base_shape": base_shape}
    return matrix, answer, meta

def reflect_symmetry_generate(rng, matrix_size=3):
    """
    Reflect symmetry across vertical axis: left and right are mirrored.
    Middle column is a transformation; last is missing.
    Simplified illustration template.
    """
    matrix = []
    size = 24
    base_shape = rng.choice(SHAPES)
    for r in range(matrix_size):
        for c in range(matrix_size):
            idx = r * matrix_size + c
            if idx == matrix_size*matrix_size - 1:
                matrix.append([])
            else:
                # left: shape at x=0.3, right mirror x=0.7, middle has two shapes
                if c == 0:
                    objs = [{"type": base_shape, "angle": 0, "size": size, "color":"black"}]
                elif c == 1:
                    objs = [{"type": base_shape, "angle": 0, "size": size, "color":"black"},
                            {"type": base_shape, "angle": 0, "size": size, "color":"black"}]
                else:
                    objs = [{"type": base_shape, "angle": 0, "size": size, "color":"black"}]
                matrix.append(objs)
    # answer mimics left column object(s)
    answer = [{"type": base_shape, "angle": 0, "size": size, "color":"black"}]
    meta = {"template":"reflect_symmetry", "base_shape": base_shape}
    return matrix, answer, meta

# registry
TEMPLATES = {
    "simple_rotate": simple_rotate_generate,
    "count_increment": count_increment_generate,
    "reflect_symmetry": reflect_symmetry_generate
}

def list_templates():
    return list(TEMPLATES.keys())
