#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#define FREQUENCY_POINTS 4096 // Replace with actual value
#define TIME_POINTS 25600     // Replace with actual value

void read_dat_to_matrix(const char *filename, int8_t **matrix, int *rows, int *cols) {
    size_t expected_size = FREQUENCY_POINTS * TIME_POINTS * sizeof(int8_t);
    FILE *fstream = fopen(filename, "rb");
    if (fstream == NULL) {
        printf("\nFile opening failed\n");
        exit(EXIT_FAILURE);
    }
    fseek(fstream, 0, SEEK_END);
    size_t file_size = ftell(fstream);
    fseek(fstream, 0, SEEK_SET);
    if (file_size != expected_size) {
        printf("Error: The file size does not match the expected matrix size.\n");
        fclose(fstream);
        exit(EXIT_FAILURE);
    }
    *matrix = (int8_t *)malloc(FREQUENCY_POINTS * TIME_POINTS * sizeof(int8_t));
    if (*matrix == NULL) {
        printf("Memory allocation failed for matrix\n");
        fclose(fstream);
        exit(EXIT_FAILURE);
    }
    fread(*matrix, sizeof(int8_t), FREQUENCY_POINTS * TIME_POINTS, fstream);
    *rows = FREQUENCY_POINTS;
    *cols = TIME_POINTS;
    fclose(fstream);
}

void write_matrix_to_csv(const char *filename, int8_t *matrix, int rows, int cols) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        perror("Error opening file for writing");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fprintf(file, "%d", matrix[i * cols + j]);
            if (j < cols - 1) {
                fprintf(file, ",");
            }
        }
        fprintf(file, "\n");
    }
    fclose(file);
}

void write_marker_matrix_to_csv(const char *filename, int8_t *matrix, int rows, int cols) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        perror("Error opening file for writing");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fprintf(file, "%d", matrix[i * cols + j]);
            if (j < cols - 1) {
                fprintf(file, ",");
            }
        }
        fprintf(file, "\n");
    }
    printf("Marker matrix saved to %s. Rows: %d, Columns: %d\n", filename, rows, cols);
    fclose(file);
}

int8_t *allocate_matrix(int rows, int cols) {
    int8_t *matrix = (int8_t *)malloc(rows * cols * sizeof(int8_t));
    if (matrix == NULL) {
        perror("Error allocating memory for matrix");
        exit(EXIT_FAILURE);
    }
    return matrix;
}

void zero_dm(int8_t *matrix, int8_t *detrended_matrix, int rows, int cols) {
    for (int freq = 0; freq < rows; freq++) {
        int sum = 0;
        for (int time = 0; time < cols; time++) {
            sum += matrix[freq * cols + time];
        }
        int mean = sum / cols;
        
        for (int time = 0; time < cols; time++) {
            detrended_matrix[freq * cols + time] = matrix[freq * cols + time] - mean;
        }
    }
}

void free_matrix(int8_t *matrix) {
    free(matrix);
}

int compare(const void *a, const void *b) {
    int8_t diff = (*(int8_t *)a - *(int8_t *)b);
    return (diff > 0) - (diff < 0); // Return 1, -1, or 0
}

int8_t calculate_median(int8_t *array, int size) {
    int8_t *temp_array = (int8_t *)malloc(size * sizeof(int8_t));
    if (!temp_array) {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < size; i++) {
        temp_array[i] = array[i];
    }

    qsort(temp_array, size, sizeof(int8_t), compare);

    int8_t median;
    if (size % 2 == 0) {
        median = (temp_array[size / 2 - 1] + temp_array[size / 2]) / 2;
    } else {
        median = temp_array[size / 2];
    }

    free(temp_array);
    return median;
}

int8_t calculate_MAD(int8_t *array, int size) {
    // Calculate the median of the original int8_t array
    int8_t median = calculate_median(array, size);

    // Create an array of absolute deviations (still using int8_t)
    int8_t *abs_deviation = (int8_t *)malloc(size * sizeof(int8_t));
    if (!abs_deviation) {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }

    // Calculate the absolute deviations from the median
    for (int i = 0; i < size; i++) {
        abs_deviation[i] = abs(array[i] - median);  // Absolute deviation from the median
    }

    // Calculate the median of absolute deviations using int8_t array
    int8_t MAD = calculate_median(abs_deviation, size);

    // Apply scaling factor for MAD
    double MAD_scaled = (double)MAD * 1.4826;

    free(abs_deviation);
    return (int8_t)MAD_scaled;  // Return the scaled MAD as int8_t
}

double apply_MAD_filter(int8_t *data, int size, int beta) {
    int8_t median = calculate_median(data, size);
    int8_t mad = calculate_MAD(data, size);
    int8_t threshold_1 = (int8_t)median + beta * mad; 
    return threshold_1;
}

int8_t *threshold_list(int8_t *data, int size, double beta, double rho, int bin_array_size, const char *output_filename) {
    double threshold_1 = apply_MAD_filter(data, size, beta);  // Result is now double
    int8_t *thresholds = (int8_t *)malloc(bin_array_size * sizeof(int8_t));
    if (thresholds == NULL) {
        printf("Memory allocation failed!\n");
        exit(EXIT_FAILURE);
    }

    FILE *file = fopen(output_filename, "w");
    if (file == NULL) {
        perror("Error opening file for writing thresholds");
        exit(EXIT_FAILURE);
    }

    fprintf(file, "Index,Threshold\n");

    for (int i = 0; i < bin_array_size; i++) {
        double calculated_threshold = threshold_1 / pow(rho, log2(i + 1));
        
        // Casting back to int8_t
        thresholds[i] = (int8_t)calculated_threshold; 
    }

    fclose(file);

    return thresholds;
}


void sumthreshold(int8_t *data, int size, double *bin_array, int8_t *thresholds, int *markers) {
    // Allocate memory for current markers
    int *current_markers = (int *)malloc(size * sizeof(int));
    if (!current_markers) {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }

    // Initialize markers to 0
    for (int i = 0; i < size; i++) {
        markers[i] = 0;
    }

    // Determine the number of iterations based on bin_array
    int max_iterations = 0;
    while (bin_array[max_iterations] > 0) {
        max_iterations++;
    }

    // Iterate through each threshold and window size
    for (int iteration = 0; iteration < max_iterations; iteration++) {
        int window_size = (int)bin_array[iteration];
        int8_t threshold = thresholds[iteration];  // Use int8_t threshold

        // Copy current markers to temporary array
        for (int i = 0; i < size; i++) {
            current_markers[i] = markers[i];
        }

        // Apply the sumthreshold algorithm to each data point
        for (int i = 0; i < size; i++) {
            int sum = 0;  // Use int for sum to avoid overflow
            int count = 0;

            // Accumulate values within the window centered around i
            for (int j = i - window_size; j <= i + window_size; j++) {
                if (j >= 0 && j < size && current_markers[j] == 0) {
                    sum += data[j];
                    count++;
                }
            }

            int8_t avg = (count > 0) ? (sum / count) : 0;  // Compute the average as int8_t

            if (avg > threshold) {
                markers[i] = 1; // Mark as flagged
                data[i] = threshold; // Replace flagged value with threshold of iteration
            }
        }
    }

    // Free allocated memory
    free(current_markers);
}

// Apply sumthreshold in time direction
void apply_sumthreshold_in_time_direction(int8_t *matrix, int rows, int cols, double *bin_array, double beta, double rho, int bin_array_size, int8_t *marker_matrix) {
    int *markers = (int *)malloc(cols * sizeof(int));  // Markers should be of type int

    for (int row = 0; row < rows; row++) {
        int8_t *signal = &matrix[row * cols];

        // Apply sumthreshold for filtering
        int8_t *thresholds = threshold_list(signal, cols, beta, rho, bin_array_size, "time_thresholds.csv");
        sumthreshold(signal, cols, bin_array, thresholds, markers);

        // Update the marker matrix and matrix itself
        for (int i = 0; i < cols; i++) {
            marker_matrix[row * cols + i] = markers[i];
            matrix[row * cols + i] = signal[i];  // Update the matrix with the filtered signal
        }

        // Free memory for thresholds
        free(thresholds);
    }

    free(markers);
}

// Apply sumthreshold in frequency direction
void apply_sumthreshold_in_frequency_direction(int8_t *matrix, int rows, int cols, double *bin_array, double beta, double rho, int bin_array_size, int8_t *marker_matrix) {
    int *markers = (int *)malloc(rows * sizeof(int));  // Markers should be of type int

    for (int col = 0; col < cols; col++) {
        int8_t *signal = (int8_t *)malloc(rows * sizeof(int8_t));

        // Initialize the signal array and markers with the previous marker array
        for (int row = 0; row < rows; row++) {
            signal[row] = matrix[row * cols + col];
            markers[row] = marker_matrix[row * cols + col];
        }

        // Apply sumthreshold for filtering
        int8_t *thresholds = threshold_list(signal, rows, beta, rho, bin_array_size, "freq_thresholds.csv");
        sumthreshold(signal, rows, bin_array, thresholds, markers);

        // Write the filtered column back to the matrix and update the marker matrix
        for (int row = 0; row < rows; row++) {
            matrix[row * cols + col] = signal[row];
            marker_matrix[row * cols + col] = markers[row]; // Update marker matrix
        }

        // Free memory for thresholds
        free(thresholds);
        free(signal);
    }

    free(markers);
}

int main() {
    const char *input_filename = "/home/ritavash/Documents/filterbank_data/output_chunks/data_part_0.dat";
    const char *output_filename_time = "final_data_time.csv";
    const char *output_filename_freq = "final_data_freq.csv";
    const char *marker_filename_time = "marker_matrix_time.csv";
    const char *marker_filename_freq = "marker_matrix_freq.csv";

    int8_t *matrix = NULL;    // Use int8_t for matrix
    int rows = 0, cols = 0;

    // Read the binary .dat file into a matrix
    read_dat_to_matrix(input_filename, &matrix, &rows, &cols);

    // Allocate memory for the final matrix and marker matrix
    int8_t *final_matrix_time = allocate_matrix(rows, cols);  // Adjusted for int8_t
    memcpy(final_matrix_time, matrix, rows * cols * sizeof(int8_t));  // Copy int8_t data

    int8_t *marker_matrix = (int8_t *)malloc(rows * cols * sizeof(int8_t));  // Change to int8_t
    memset(marker_matrix, 0, rows * cols * sizeof(int8_t)); // Initialize marker matrix to 0

    // Allocate memory for detrended matrix
    int8_t *detrended_matrix = (int8_t *)malloc(rows * cols * sizeof(int8_t));
    if (detrended_matrix == NULL) {
        perror("Error allocating memory for detrended_matrix");
        free_matrix(matrix);
        free_matrix(final_matrix_time);
        free(marker_matrix);
        exit(EXIT_FAILURE);
    }

    // Apply zero DM (detrending) to the matrix
    zero_dm(matrix, detrended_matrix, rows, cols);

    // Copy detrended data into the final matrix
    memcpy(final_matrix_time, detrended_matrix, rows * cols * sizeof(int8_t));

    // Define parameters for sumthreshold
    double bin_array[] = {1, 2, 4, 0}; // Example bin sizes
    double beta = 1.0;              // Threshold scaling factor
    double rho = 1.5;               // Exponential decay factor
    int bin_array_size = sizeof(bin_array) / sizeof(bin_array[0]);

    // Apply sumthreshold in time direction and save the result
    apply_sumthreshold_in_time_direction(final_matrix_time, rows, cols, bin_array, beta, rho, bin_array_size, marker_matrix);
    write_matrix_to_csv(output_filename_time, final_matrix_time, rows, cols);  // Save filtered matrix

    // Save the marker matrix after time run
    write_marker_matrix_to_csv(marker_filename_time, marker_matrix, rows, cols);

    // Apply sumthreshold in frequency direction and save the result
    apply_sumthreshold_in_frequency_direction(final_matrix_time, rows, cols, bin_array, beta, rho, bin_array_size, marker_matrix);
    write_matrix_to_csv(output_filename_freq, final_matrix_time, rows, cols);  // Save filtered matrix

    // Save the marker matrix after frequency run
    write_marker_matrix_to_csv(marker_filename_freq, marker_matrix, rows, cols);

    // Free memory for allocated matrices
    free_matrix(matrix);
    free_matrix(final_matrix_time);
    free(marker_matrix);
    free(detrended_matrix);  // Free the detrended matrix

    return 0;
}