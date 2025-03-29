import numpy as np
from matplotlib import pyplot

class Fitter:
    IMAGE_SET_COUNT = 3
    IMAGE_COUNT = 5
    IMAGE_X_SIZE = 24
    IMAGE_Y_SIZE = 24

    SOLUTION_COUNT = 1000
    PATTERN_COUNT = 7
    PATTERN_X_SIZE = 3
    PATTERN_Y_SIZE = 3
    ITERATION_COUNT = 100

    def __init__(self):
        pass


    def load_images(self):
        images = np.zeros((Fitter.IMAGE_SET_COUNT * Fitter.IMAGE_COUNT,
                           Fitter.IMAGE_X_SIZE, Fitter.IMAGE_Y_SIZE))
        for image_index in range(Fitter.IMAGE_SET_COUNT * Fitter.IMAGE_COUNT):
            image_file_name = f"{image_index}.bin"
            with open(image_file_name, 'r') as image_file:
                bit_string = image_file.read()
            bit_string = bit_string.replace("\n", "")
            for i in range(len(bit_string)):
                images[image_index][i // Fitter.IMAGE_X_SIZE \
                    ][i % Fitter.IMAGE_Y_SIZE] = \
                    int(bit_string[i])

        return images

    def place_most_similar_pattern(self, solution, 
                                   approximate_image,
                                   current_image,
                                   x_index,
                                   y_index):
        raise NotImplementedError

    
    def construct_approximate_image(self, image_set_index,
                                    images, solution,
                                    image_index): 
        approximate_image = np.zeros(
            (Fitter.IMAGE_X_SIZE, Fitter.IMAGE_Y_SIZE))
        current_image_index = \
            Fitter.IMAGE_COUNT * image_set_index + \
            image_index
        current_image = \
            images[current_image_index]

        for x_index in range(
            0, Fitter.IMAGE_X_SIZE, Fitter.PATTERN_X_SIZE):
            for y_index in range(
                0, Fitter.IMAGE_Y_SIZE,
                Fitter.PATTERN_Y_SIZE):
                self.place_most_similar_pattern(
                    solution, approximate_image, 
                    current_image, x_index, y_index 
                )

        return approximate_image


    def compute_loss(self, image_set_index, images,
                     solution):
        loss = 0

        for image_index in range(Fitter.IMAGE_COUNT):
            approximate_image = \
                self.construct_approximate_image(
                    image_set_index, images,
                    solution, image_index
                )
        
        return loss

        
    def compute_losses(self, image_set_index,
                       images, solutions):
        losses = np.zeros((Fitter.SOLUTION_COUNT,))
        
        for solution_index in range(Fitter.SOLUTION_COUNT):
            losses[solution_index] = self.compute_loss(
                image_set_index, images,
                solutions[solution_index])

        return losses


    def fit_to_image_set(self, image_set_index, images):
        print(f"Image set: {image_set_index}")

        solutions = (np.random.rand(
            Fitter.SOLUTION_COUNT, Fitter.PATTERN_COUNT,
            Fitter.PATTERN_X_SIZE, Fitter.PATTERN_Y_SIZE) > 0.5
            ).astype(int)

        for current_iteration in range(Fitter.ITERATION_COUNT):
            losses = self.compute_losses(image_set_index, images, solutions)


    def run(self):
        images = self.load_images()

        for image_set_index in range(Fitter.IMAGE_SET_COUNT):
            self.fit_to_image_set(image_set_index, images)

def main():
    Fitter().run()


if __name__ == "__main__":
    main()
