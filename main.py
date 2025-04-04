import numpy as np
from matplotlib import pyplot

class Fitter:
    IMAGE_SET_COUNT = 3
    IMAGE_COUNT = 5
    IMAGE_X_SIZE = 24
    IMAGE_Y_SIZE = 24

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
                    ][i % Fitter.IMAGE_Y_SIZE] = int(bit_string[i])
        return images

    def place_most_similar_pattern(self, solution, 
                                   approximate_image,
                                   current_image,
                                   x_index,
                                   y_index):
        block = current_image[x_index:x_index + Fitter.PATTERN_X_SIZE,
                              y_index:y_index + Fitter.PATTERN_Y_SIZE]
        min_diff = float('inf')
        best_pattern = None
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
        approximate_image = np.zeros((Fitter.IMAGE_X_SIZE, Fitter.IMAGE_Y_SIZE))
        current_image_index = Fitter.IMAGE_COUNT * image_set_index + image_index
        current_image = images[current_image_index]

        for x_index in range(0, Fitter.IMAGE_X_SIZE, Fitter.PATTERN_X_SIZE):
            for y_index in range(0, Fitter.IMAGE_Y_SIZE, Fitter.PATTERN_Y_SIZE):
                self.place_most_similar_pattern(solution, approximate_image, current_image, x_index, y_index)
        return approximate_image

    def compute_loss(self, image_set_index, images, solution):
        loss = 0
        for image_index in range(Fitter.IMAGE_COUNT):
            approximate_image = self.construct_approximate_image(image_set_index, images, solution, image_index)
            original_image = images[Fitter.IMAGE_COUNT * image_set_index + image_index]
            loss += np.sum(np.abs(approximate_image - original_image))
        return loss

    def compute_losses(self, image_set_index, images, solutions):
        num_solutions = solutions.shape[0]
        losses = np.zeros((num_solutions,))
        for solution_index in range(num_solutions):
            losses[solution_index] = self.compute_loss(image_set_index, images, solutions[solution_index])
        return losses

    def fit_to_image_set(self, image_set_index, images, mutation_rate=0.01, solution_count=1000):
        print(f"Image set: {image_set_index}")

        population_size = solution_count
        elite_count = int(0.05 * population_size)

        solutions = (np.random.rand(
            population_size, Fitter.PATTERN_COUNT,
            Fitter.PATTERN_X_SIZE, Fitter.PATTERN_Y_SIZE) > 0.5).astype(int)

        best_losses = []

        for current_iteration in range(Fitter.ITERATION_COUNT):
            losses = self.compute_losses(image_set_index, images, solutions)
            sorted_indices = np.argsort(losses)
            solutions = solutions[sorted_indices]

            best_loss = losses[sorted_indices[0]]
            avg_loss = np.mean(losses)
            best_losses.append(best_loss)

            print(f"Iteration {current_iteration + 1}, Best loss: {best_loss}, Avg loss: {avg_loss:.2f}")

            new_solutions = solutions[:elite_count].copy()

            while len(new_solutions) < population_size:
                parents = solutions[np.random.choice(elite_count, size=2, replace=False)]
                crossover_point = np.random.randint(1, Fitter.PATTERN_COUNT)
                child = np.concatenate([
                    parents[0][:crossover_point],
                    parents[1][crossover_point:]
                ], axis=0)
                mutation_mask = (np.random.rand(*child.shape) < mutation_rate).astype(int)
                child = np.bitwise_xor(child, mutation_mask)
                new_solutions = np.concatenate([new_solutions, [child]], axis=0)

            solutions = new_solutions[:population_size]

            # Her 3 iterasyonda bir reconstructed image göster
            if (current_iteration + 1) % 3 == 0:
                temp_best = solutions[0]
                approx_image = self.construct_approximate_image(image_set_index, images, temp_best, 0)
                pyplot.imshow(approx_image, cmap='gray')
                pyplot.title(f"Approx Image - Iter {current_iteration + 1} (Set {image_set_index})")
                pyplot.axis('off')
                pyplot.savefig(f"approx_iter_{current_iteration+1}_set_{image_set_index}_mut_{mutation_rate}_pop_{solution_count}.png")
                pyplot.close()

        # Loss trend grafiği
        pyplot.plot(best_losses, label='Best Loss')
        pyplot.xlabel("Iteration")
        pyplot.ylabel("Loss")
        pyplot.title(f"Image Set {image_set_index} - Loss (mut={mutation_rate}, pop={solution_count})")
        pyplot.legend()
        pyplot.grid(True)
        pyplot.savefig(f"loss_plot_set_{image_set_index}_mut_{mutation_rate}_pop_{solution_count}.png")
        pyplot.close()

        # En iyi çözüm
        best_solution = solutions[0]
        print(f"\nBest 7 patterns for image set {image_set_index}:")
        for idx, pattern in enumerate(best_solution):
            print(f"\nPattern {idx + 1}:\n{pattern}")

        with open(f"patterns_image_set_{image_set_index}_mut_{mutation_rate}_pop_{solution_count}.txt", "w") as f:
            for idx, pattern in enumerate(best_solution):
                f.write(f"Pattern {idx + 1}:\n{pattern}\n\n")

        approx_image = self.construct_approximate_image(image_set_index, images, best_solution, 0)
        pyplot.imshow(approx_image, cmap='gray')
        pyplot.title(f"Reconstructed Image (Set {image_set_index})")
        pyplot.axis('off')
        pyplot.savefig(f"final_reconstructed_set_{image_set_index}_mut_{mutation_rate}_pop_{solution_count}.png")
        pyplot.close()

    def run(self):
        images = self.load_images()
        mutation_rates = [0.01]
        solution_counts = [500, 1000, 2000]

        for mut in mutation_rates:
            for sol_count in solution_counts:
                print(f"\n>>> Mutation rate: {mut}, Solution count: {sol_count}")
                self.fit_to_image_set(2, images, mutation_rate=mut, solution_count=sol_count)

def main():
    Fitter().run()

if __name__ == "__main__":
    main()