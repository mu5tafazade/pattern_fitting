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
    ITERATION_COUNT = 15

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
        # Take 3x3 block to fill with the most similar pattern
        block = current_image[x_index:x_index + Fitter.PATTERN_X_SIZE,
                              y_index:y_index + Fitter.PATTERN_Y_SIZE]

        # Find the pattern with the minimum difference

        # Minimum difference
        min_diff = float('inf')

        # Pattern with the minimum difference
        best_pattern = None

        # TODO: add comments

        for pattern_index in range(Fitter.PATTERN_COUNT):
            pattern = solution[pattern_index]
            diff = np.sum(np.abs(block - pattern))
            if diff < min_diff:
                min_diff = diff
                best_pattern = pattern

        approximate_image[x_index:x_index + Fitter.PATTERN_X_SIZE,
                          y_index:y_index + Fitter.PATTERN_Y_SIZE] = best_pattern

    def construct_approximate_image(self, image_set_index,
                                    images, solution,
                                    image_index): 
        approximate_image = np.zeros(
            (Fitter.IMAGE_X_SIZE, Fitter.IMAGE_Y_SIZE))
        current_image_index = \
            Fitter.IMAGE_COUNT * image_set_index + \
            image_index
        current_image = images[current_image_index]

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

    def compute_loss(self, image_set_index, images, solution):
        loss = 0
        for image_index in range(Fitter.IMAGE_COUNT):
            approximate_image = self.construct_approximate_image(
                image_set_index, images, solution, image_index
            )
            original_image = images[Fitter.IMAGE_COUNT * image_set_index + image_index]
            loss += np.sum(np.abs(approximate_image - original_image))

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

        population_size = Fitter.SOLUTION_COUNT

        # Number of best solutions
        elite_count = 50

        # Bit flip with 1% probability
        mutation_rate = 0.01

        # Initial population
        solutions = (np.random.rand(
            population_size, Fitter.PATTERN_COUNT,
            Fitter.PATTERN_X_SIZE, Fitter.PATTERN_Y_SIZE) > 0.5).astype(int)

        # Track best losses for plotting
        best_losses = []

        for current_iteration in range(Fitter.ITERATION_COUNT):
            losses = self.compute_losses(image_set_index, images, solutions)

            # Sort solutions with losses in ascending order
            sorted_indices = np.argsort(losses)
            solutions = solutions[sorted_indices]

            best_loss = losses[sorted_indices[0]]
            avg_loss = np.mean(losses)
            best_losses.append(best_loss)

            print(f"Iteration {current_iteration + 1}, Best loss: {best_loss}, Avg loss: {avg_loss:.2f}")

            # Copy best solutions to the new population
            new_solutions = solutions[:elite_count].copy()

            # Generate new solutions
            # Total solution count will not change
            while len(new_solutions) < population_size:
                # Select 2 random elite solutions
                parents = solutions[np.random.choice(elite_count, size=2, replace=False)]

                # Apply crossover using single middle point
                crossover_point = np.random.randint(1, Fitter.PATTERN_COUNT)
                child = np.concatenate([
                    parents[0][:crossover_point],
                    parents[1][crossover_point:]
                ], axis=0)

                # Apply mutation with small probability
                mutation_mask = (np.random.rand(*child.shape) < mutation_rate).astype(int)
                child = np.bitwise_xor(child, mutation_mask)

                # Add the new child to the new population
                new_solutions = np.concatenate([new_solutions, [child]], axis=0)

            solutions = new_solutions[:population_size]

        # Plot the change of loss values with respect to iterations
        pyplot.plot(best_losses, label='Best Loss')
        pyplot.xlabel("Iteration")
        pyplot.ylabel("Loss")
        pyplot.title(f"Image Set {image_set_index} - Best Loss Over Time")
        pyplot.legend()
        pyplot.grid(True)
        pyplot.show()

    def run(self):
        images = self.load_images()

        for image_set_index in range(Fitter.IMAGE_SET_COUNT):
            self.fit_to_image_set(image_set_index, images)

def main():
    Fitter().run()

if __name__ == "__main__":
    main()
