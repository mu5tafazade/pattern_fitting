import numpy as np


def main():
    image_set_count = 3
    image_count = 5
    image_x_size = 24
    image_y_size = 24
    solution_count = 1000

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
                    if current_iteration == 0 and \
                        solution_index == 0:
                        print(current_image_index)





if __name__ == "__main__":
    main()
