import numpy as np


def main():
    image_set_count = 3
    image_count = 5
    image_x_size = 24
    image_y_size = 24
    solution_count = 1000

    images = np.zeros((image_set_count * image_count,
                       image_x_size, image_y_size))
    for image_index in range(image_set_count * image_count):
        image_file_name = f"{image_index}.bin"
        with open(image_file_name, 'r') as image_file:
            bit_string = image_file.read()
        bit_string = bit_string.replace("\n", "")
        for i in range(len(bit_string)):
            images[image_index][i // image_x_size \
                ][i % image_y_size] = \
                int(bit_string[i])

    for image_set_index in range(image_set_count):
        solutions = (np.random.rand(
            solution_count, 7, 3, 3) > 0.5).astype(int)

        losses = np.zeros((solution_count,))

        iteration_count = 100

        for current_iteration in range(iteration_count):
            for solution_index in range(solution_count):
                losses[solution_index] = 0 
                for image_index in range(image_count):
                    approximate_image = np.zeros(
                        (image_x_size, image_y_size))
                    current_image_index = \
                        image_count * image_set_index + \
                        image_index

if __name__ == "__main__":
    main()
