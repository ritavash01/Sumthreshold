#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>
//Define the macros 
#define FREQUENCY_POINTS 4096 // Replace with actual value
#define TIME_POINTS 128000     // Replace with actual value
#define THRESHOLD 3  // Outlier threshold (integer approximation)
#define ALPHA 128     // Alpha factor (fixed-point: 128 = 0.5 in Q8 format)
#define SUBGROUP_SIZE 160
#define THRESHOLD_PERCENT 90 // 70% of 16 means at least 12 zeros
// #define OvERLAP 64 //overlap between each subgroup 
// #define STRIDE 64 // SUBGROUP_SIZE - OVERLAP
/*
This section of the code deals with reading the data. 
The data (intensity) is signed 8 bit integer data 
From a dat file, we have 4096 (Frequency channels) rows and 25600 columns (time channels), 
hence the size of the data is 4096*25600*size(int8_t)
The function returns a int8 matrix, which can be used for futher processing
*/

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

/*
The following fucntion generates a mask matrix as the same size as the data
USAGE: int8_t *mask = create_mask_matrix(FREQUENCY_POINTS, TIME_POINTS);
*/

int8_t *create_mask_matrix(int rows, int cols) {
    int8_t *mask = (int8_t *)malloc(rows * cols * sizeof(int8_t));
    if (mask == NULL) {
        printf("Memory allocation failed for mask matrix\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < rows * cols; i++) {
        mask[i] = 1;
    }
    return mask;
}

/*
The function can downsample the rows of any matrix by a given factor
pointing to the correct matrix and providing the originla number of rows, number of columns, factor,
and the number of new rows will do the trick 
*/


int8_t *downsample_rows(const int8_t *matrix, int original_rows, int cols, int factor, int *new_rows) {
    if (factor <= 0 || original_rows % factor != 0) {
        printf("Invalid downsampling factor. It should evenly divide the number of rows.\n");
        exit(EXIT_FAILURE);
    }

    *new_rows = original_rows / factor;
    int8_t *downsampled_matrix = (int8_t *)malloc((*new_rows) * cols * sizeof(int8_t));
    if (downsampled_matrix == NULL) {
        printf("Memory allocation failed for downsampled matrix\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < *new_rows; i++) {
        for (int j = 0; j < cols; j++) {
            int sum = 0;
            for (int k = 0; k < factor; k++) {
                sum += matrix[(i * factor + k) * cols + j]; // Sum over 'factor' rows
            }
            downsampled_matrix[i * cols + j] = (int8_t)(sum / factor); // Compute mean
        }
    }

    return downsampled_matrix;
}



/*
This section calculates the zscore
*/
// Function to compare int8_t values (for qsort)
int compare(const void *a, const void *b) {
    return (*(int8_t*)a - *(int8_t*)b);
}

// Function to compute median for int8_t arrays without modifying the original array
int8_t median_calc(const int8_t* arr, int n) {
    // Allocate a temporary array to hold a copy of the data
    int8_t *sorted = (int8_t*)malloc(n * sizeof(int8_t));
    if (!sorted) {
        printf("Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    memcpy(sorted, arr, n * sizeof(int8_t));
    
    // Sort the copy
    qsort(sorted, n, sizeof(int8_t), compare);
    int8_t median = sorted[n / 2];
    free(sorted);
    return median;
}

// Function to compute MAD (Median Absolute Deviation) without modifying the original array
int8_t mad_calc(const int8_t* arr, int n, int8_t median) {
    int8_t *devs = (int8_t *)malloc(n * sizeof(int8_t));
    if (!devs) {
        printf("Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < n; i++)
        devs[i] = abs(arr[i] - median);

    // Use the modified median_calc that doesn't sort in place
    int8_t mad = median_calc(devs, n);
    free(devs);
    return mad;
}

int8_t stddev_calc(const int8_t *data, int size, double mean) {
    double sum_sq_diff = 0.0;

    for (int i = 0; i < size; i++) {
        int8_t diff = data[i] - mean;
        sum_sq_diff += diff * diff;
    }

    return sqrt(sum_sq_diff / size);  // Population standard deviation
}



// Approximate modified Z-score using int8_t math
int8_t modified_zscore(int8_t val, int8_t median, int8_t mad) {
    if (mad == 0) return (val == median) ? 0 : 100;  // Large value for outliers
    return abs(val - median) / mad;  // Approximate integer Z-score
}

void detect_outliers(int8_t* matrix, int8_t* mask, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        int8_t *transit_score = (int8_t *)malloc(cols * sizeof(int8_t));
        if (!transit_score) {
            printf("Memory allocation failed\n");
            exit(EXIT_FAILURE);
        }

        // Initialize transit scores
        transit_score[0] = 0;  // Start with zero for first time step

        for (int j = 1; j < cols; j++) {
            int8_t int0 = abs(matrix[i * cols + (j - 1)]);
            int8_t int1 = abs(matrix[i * cols + j]);
            int8_t rate_of_change = abs(int1 - int0);

            // Apply exponential smoothing
            transit_score[j] = ((ALPHA * rate_of_change) >> 8) + (((256 - ALPHA) * transit_score[j - 1]) >> 8);
        }

        // Compute median & MAD using copies (so transit_score remains in original order)
        int8_t median_val = median_calc(transit_score, cols);
        int8_t mad_val = mad_calc(transit_score, cols, median_val);

        // Detect outliers
        for (int j = 0; j < cols; j++) {
            int8_t zscore = modified_zscore(transit_score[j], median_val, mad_val);
            if (zscore > THRESHOLD)
                mask[i * cols + j] = 0;  // Mark as outlier
        }

        free(transit_score);
    }
}


void restore_mask(int8_t* mask, int rows, int cols) {
    const int TOTAL_ROWS = 1024;
//     for (int c = 0; c < cols; c++) {
//         for (int start = 0; start < rows; start += STRIDE) {
//             int zero_count = 0;
//             int end = start + SUBGROUP_SIZE;
//             if (end > rows) {
//                 end = rows;  // Ensure we don't go out of bounds
//             }

//             // Count the number of zeros in the subgroup
//             for (int r = start; r < end; r++) {
//                 if (mask[r * cols + c] == 0) {
//                     zero_count++;
//                 }
//             }

//             // If zeros < threshold, restore all to 1
//             if (zero_count < (THRESHOLD_PERCENT * (end - start)) / 100) {
//                 for (int r = start; r < end; r++) {
//                     mask[r * cols + c] = 1;
//                 }
//             }
//         }
//     }
// }
         // Ensure the total number of groups is correct (1024 groups, so total rows should be 96 * 1024 = 98304)
    if (rows != TOTAL_ROWS) {
        printf("Warning: The number of rows does not match the expected 1024 rows.\n");
        return;
    }

    // Iterate over the columns
    for (int c = 0; c < cols; c++) {
        // Process the rows in subgroups of size 96
        for (int start = 0; start < rows; start += SUBGROUP_SIZE) {
            int subgroup_size = SUBGROUP_SIZE;

            // If we are at the last subgroup and it's smaller than 96, adjust the size
            if (start + SUBGROUP_SIZE > rows) {
                subgroup_size = rows - start;
            }

            int zero_count = 0;

            // Count the number of zeros in the current subgroup
            for (int r = start; r < start + subgroup_size; r++) {
                if (mask[r * cols + c] == 0) {
                    zero_count++;
                }
            }

            // If the number of zeros is below the threshold, restore the entire subgroup to 1
            if (zero_count < (THRESHOLD_PERCENT * subgroup_size) / 100) {
                for (int r = start; r < start + subgroup_size; r++) {
                    mask[r * cols + c] = 1;
                }
            }
        }
    }
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

void write_matrix_to_dat(const char *filename, int8_t *matrix, int rows, int cols) {
    FILE *file = fopen(filename, "wb");
    if (file == NULL) {
        perror("Error opening file for writing");
        exit(EXIT_FAILURE);
    }

    // Write rows and columns as metadata (4 bytes each, assuming int)
    fwrite(&rows, sizeof(int), 1, file);
    fwrite(&cols, sizeof(int), 1, file);

    // Write matrix data
    fwrite(matrix, sizeof(int8_t), rows * cols, file);

    printf("Matrix saved to %s. Rows: %d, Columns: %d\n", filename, rows, cols);
    fclose(file);
}

/*
This part of the code does the detrending of the data before narrowband RFI removal
*/

int rounded_division(int numerator, int denominator) {
    // Helper function to perform rounded division.
    return (numerator + denominator / 2) / denominator;
}
/*
This section deals with the zero DMing of data, I am also including Z dot calculation 
*/
void remove_column_and_row_mean_int8(int8_t* matrix, int rows, int cols) {
    // Remove column mean first
    // for (int j = 0; j < cols; j++) {
    //     int sum = 0;
    //     for (int i = 0; i < rows; i++) {
    //         sum += matrix[i * cols + j];
    //     }
    //     int mean = rounded_division(sum, rows);
    //     for (int i = 0; i < rows; i++) {
    //         int temp = matrix[i * cols + j] - mean;
    //         // Ensure the result stays in int8_t range (-128 to 127)
    //         if (temp > 127) temp = 127;
    //         if (temp < -128) temp = -128;
    //         matrix[i * cols + j] = (int8_t)temp;
    //     }
    // }


    // Remove row mean next
    for (int i = 0; i < rows; i++) {
        int sum = 0;
        for (int j = 0; j < cols; j++) {
            sum += matrix[i * cols + j];
        }
        int mean = rounded_division(sum, cols);
        for (int j = 0; j < cols; j++) {
            int temp = matrix[i * cols + j] - mean;
            if (temp > 127) temp = 127;
            if (temp < -128) temp = -128;
            matrix[i * cols + j] = (int8_t)temp;
        }
    }
}
// Function to calculate alpha_i
float calculate_alpha(int8_t* ti, int8_t* s_dm0, int N) {
    int64_t sum_ti = 0, sum_sdm0 = 0, sum_ti_sdm0 = 0, sum_sdm0_sq = 0;
    
    #pragma omp parallel for reduction(+:sum_ti, sum_sdm0, sum_ti_sdm0, sum_sdm0_sq)
    for (int k = 0; k < N; k++) {
        sum_ti += ti[k];
        sum_sdm0 += s_dm0[k];
        sum_ti_sdm0 += ti[k] * s_dm0[k];
        sum_sdm0_sq += s_dm0[k] * s_dm0[k];
    }
    
    float num = (float)(sum_ti_sdm0 - (1.0 / N) * sum_ti * sum_sdm0);
    float den = (float)(sum_sdm0_sq - (1.0 / N) * sum_sdm0 * sum_sdm0);
    
    return (den != 0) ? num / den : 0.0f;  // Avoid division by zero
}

// Z-dot filter implementation
void apply_zdot_filter(int8_t* matrix, int rows, int cols) {
    // Compute zero-DM time series s_dm0 (mean across all channels)
    int8_t* s_dm0 = (int8_t*)calloc(rows, sizeof(int8_t));
    
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        int sum = 0;
        for (int j = 0; j < cols; j++) {
            sum += matrix[i * cols + j];
        }
        s_dm0[i] = sum / cols;  // Compute mean per time sample
    }
    
    // Apply Z-dot filter to each channel
    #pragma omp parallel for
    for (int j = 0; j < cols; j++) {
        int8_t* ti = (int8_t*)malloc(rows * sizeof(int8_t));
        
        // Copy time series for channel j
        for (int i = 0; i < rows; i++) {
            ti[i] = matrix[i * cols + j];
        }
        
        // Calculate alpha for this channel
        float alpha = calculate_alpha(ti, s_dm0, rows);
        
        // Filter the time series t' = ti - alpha * s_dm0
        for (int i = 0; i < rows; i++) {
            int temp = ti[i] - (int)(alpha * s_dm0[i]);
            if (temp > 127) temp = 127;
            if (temp < -128) temp = -128;
            matrix[i * cols + j] = (int8_t)temp;
        }
        
        free(ti);
    }
    
    free(s_dm0);
}


// Function to calculate the median of the matrix
int8_t compute_median(int8_t* matrix, int total_elements) {
    int8_t* temp = (int8_t*)malloc(total_elements * sizeof(int8_t));
    if (!temp) {
        printf("Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Copy values into temp array
    for (int i = 0; i < total_elements; i++) {
        temp[i] = matrix[i];
    }

    // Sort the array
    qsort(temp, total_elements, sizeof(int8_t), compare);

    // Compute median
    int8_t median;
    if (total_elements % 2 == 0)
        median = (temp[total_elements / 2 - 1] + temp[total_elements / 2]) / 2;
    else
        median = temp[total_elements / 2];

    free(temp);
    return median;
}

void replace_masked_values(int8_t* matrix, int8_t* mask, int rows, int cols) {
    int total_elements = rows * cols;
    int8_t median = compute_median(matrix, total_elements);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (mask[i * cols + j] == 0) {  // If marked as 0 in mask
                matrix[i * cols + j] = median;
            }
        }
    }
}

void update_mask_with_zscore(int8_t* matrix, int8_t* mask, int rows, int cols, int8_t threshold) {
    for (int i = 0; i < rows; i++) {
        // Compute median and MAD for the row
        int8_t median_val = median_calc(&matrix[i * cols], cols);
        int8_t mad_val = mad_calc(&matrix[i * cols], cols, median_val);

        // Compute modified Z-score and update mask
        for (int j = 0; j < cols; j++) {
            int8_t zscore = modified_zscore(matrix[i * cols + j], median_val, mad_val);
            if (zscore > threshold) {
                mask[i * cols + j] = 0;  // Mark as an outlier
            }
        }
    }
}

void generate_new_mask_and_union(const int8_t* matrix, int8_t* new_mask, int8_t* previous_mask, int rows, int cols, int8_t zscore_threshold, int window_size, float zero_threshold) {
    int zero_limit = (int)(zero_threshold * window_size);  // Compute threshold count for zeros
    int8_t* row_medians = (int8_t*)malloc(rows * sizeof(int8_t));
    int8_t* row_mask = (int8_t*)malloc(rows * sizeof(int8_t));  // Mask status column for each row
    if (!row_medians || !row_mask) {
        printf("Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Step 1: Compute median for each row and initialize row_mask
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        row_medians[i] = median_calc(&matrix[i * cols], cols);
        row_mask[i] = 1;  // Initially consider all rows unmasked
    }

    // Step 2: Compute global median of row medians
    int8_t global_median = median_calc(row_medians, rows);

    // Step 3: Compute standard deviation of row medians
    int8_t sigma = stddev_calc(row_medians, rows, global_median);

    // Step 4: Mask entire row if its median > (global_median + 3σ)
    memset(new_mask, 1, rows * cols * sizeof(int8_t));  // Initialize mask to 1

    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        if (row_medians[i] > global_median + 3 * sigma) {
            row_mask[i] = 0;  // Mark the row as masked
            for (int j = 0; j < cols; j++) {
                new_mask[i * cols + j] = 0;  // Mask entire row
            }
        }
    }

    free(row_medians);  // Cleanup row medians

    // Step 5: Apply Z-score thresholding (skip masked rows)
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        if (row_mask[i] == 0) continue;  // Skip masked rows

        int8_t median_val = median_calc(&matrix[i * cols], cols);
        int8_t mad_val = mad_calc(&matrix[i * cols], cols, median_val);

        for (int j = 0; j < cols; j++) {
            double zscore = modified_zscore(matrix[i * cols + j], median_val, mad_val);
            if (zscore > zscore_threshold) {
                new_mask[i * cols + j] = 0;  // Mark as an outlier
            }
        }
    }

    // Step 6: Apply window-based restoration (using row_mask to skip masked rows)
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        if (row_mask[i] == 0) continue;  // Skip masked rows

        for (int j = 0; j <= cols - window_size; j++) {
            int zero_count = 0;

            // Count the number of zeros in the current window
            for (int k = j; k < j + window_size; k++) {
                if (new_mask[i * cols + k] == 0) {
                    zero_count++;
                }
            }

            // If zero count is below threshold, restore those positions to 1
            if (zero_count < zero_limit) {
                for (int k = j; k < j + window_size; k++) {
                    new_mask[i * cols + k] = 1;
                }
            }
        }
    }

    // Step 7: Take intersection of new_mask and previous_mask (bitwise AND)
    #pragma omp parallel for
    for (int i = 0; i < rows * cols; i++) {
        new_mask[i] = new_mask[i] & previous_mask[i];  // Intersection by ANDing
    }

    free(row_mask);  // Cleanup row_mask
}



void upsample_mask(const int8_t *mask_downsampled, int8_t *mask_upsampled, int old_rows, int new_rows, int cols) {
    int factor = new_rows / old_rows;  // Should be 4096 / 1024 = 4

    for (int i = 0; i < old_rows; i++) {
        for (int j = 0; j < factor; j++) {
            int new_index = i * factor + j;
            for (int k = 0; k < cols; k++) {
                mask_upsampled[new_index * cols + k] = mask_downsampled[i * cols + k];
            }
        }
    }
}




int main() {
    const char *filename = "/home/ritavash/Documents/filterbank_data/B0329_chunks/chunk_0.dat";  // Replace with actual filename
    // const char *filename = "/home/ritavash/Documents/filterbank_data/output_chunks/data_part_0.dat";
    int8_t *matrix;
    int rows, cols;

    // Step 1: Read data from file and create the mask
    read_dat_to_matrix(filename, &matrix, &rows, &cols);
    int8_t *mask = create_mask_matrix(rows, cols);

    // Step 2: Downsample the matrix and the mask
    int downsample_factor = 4;  // Adjust as needed
    int new_rows;
    int8_t *downsampled_matrix = downsample_rows(matrix, rows, cols, downsample_factor, &new_rows);
    int8_t *downsampled_mask = downsample_rows(mask, rows, cols, downsample_factor, &new_rows);

    // Step 3: Detect outliers and restore mask on downsampled data
    detect_outliers(downsampled_matrix, downsampled_mask, new_rows, cols);
    restore_mask(downsampled_mask, new_rows, cols);
    // write_marker_matrix_to_csv(".csv", downsampled_mask, new_rows, cols);
    // Step 4: Write final downsampled mask to CSV
    

    // remove_column_and_row_mean_int8(downsampled_matrix, new_rows, cols);
    // write_marker_matrix_to_csv("zero_dm_downsampled_matrix.csv", downsampled_matrix, new_rows, cols);
    // apply_zdot_filter(downsampled_matrix, new_rows, cols);
    // write_marker_matrix_to_csv("zdot_downsampled_matrix.csv", downsampled_matrix, new_rows, cols);
    replace_masked_values(downsampled_matrix, downsampled_mask, new_rows, cols);
    // write_marker_matrix_to_csv("downsampled_matrix.csv", downsampled_matrix, new_rows, cols);
    int8_t *new_mask = (int8_t*)malloc(new_rows * cols * sizeof(int8_t));
    if (!new_mask) {
        printf("Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    generate_new_mask_and_union(downsampled_matrix, new_mask, downsampled_mask, new_rows, cols, 3, 100, 0.5);
    
    

    //Upsampling the mask back to 4096

    int upsampled_rows = new_rows * 4;  // Since we downsampled by factor of 4
    int8_t *upsampled_mask = (int8_t*)malloc(upsampled_rows * cols * sizeof(int8_t));
    if (!upsampled_mask) {
        printf("Memory allocation failed for upsampled mask\n");
        exit(EXIT_FAILURE);
    }
    upsample_mask(new_mask, upsampled_mask, new_rows, upsampled_rows, cols);
    
    write_marker_matrix_to_csv("final_upsampled_mask.csv", upsampled_mask, upsampled_rows, cols);
    // remove_column_and_row_mean_int8(matrix, rows, cols);
    replace_masked_values(matrix, upsampled_mask, rows, cols);
    apply_zdot_filter(matrix, rows, cols);
    
    // write_marker_matrix_to_csv("final_upsampled_matrix_zero_dm.csv", matrix, upsampled_rows, cols);
    write_matrix_to_dat("rfi_chunk.dat", matrix, upsampled_rows, cols);
    // Cleanup memory
    free(matrix);
    free(mask);
    free(downsampled_matrix);
    free(downsampled_mask);
    free(new_mask);
    free(upsampled_mask);
    return 0;
}
