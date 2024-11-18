import os
import argparse
import numpy as np
from tqdm import tqdm

"""
https://github.com/ultralytics/ultralytics/issues/3592
"""

def process_file(filename):
    segments = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip().split(' ')
            cls = int(line[0])
            x, y, w, h = map(float, line[1:])
            x_min = x - (w / 2)
            y_min = y - (h / 2)
            x_max = x + (w / 2)
            y_max = y + (h / 2)
            segment = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
            segments.append(segment)
    return cls, segments

def save_segments(filename, cls, segments):
    """
    Save the segments to a file.

    Args:
        filename (str): path to the output file.
        cls (int): class integer
        segments (list): list of segments, each segment is a list of points, each point is a list of x, y coordinates.
    """
    with open(filename, 'w') as file:
        for segment in segments:
            line = f'{cls} ' + ' '.join([f'{coord[0]:.6f},{coord[1]:.6f}' for coord in segment])
            file.write(line + '\n')

def main():
    parser = argparse.ArgumentParser(description='Convert bounding boxes to segmentation points.')
    parser.add_argument('--input_dir', type=str, help='Path to the directory containing input files.')
    parser.add_argument('--output_dir', type=str, help='Path to the directory to save the output files.')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    # Iterate over the files in the input directory
    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith('.txt'):
            filepath = os.path.join(input_dir, filename)
            cls, segments = process_file(filepath)
            new_filepath = os.path.join(output_dir, filename)
            save_segments(new_filepath, cls, segments)

if __name__ == '__main__':
    main()