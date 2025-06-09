#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h> // برای اندازه‌گیری زمان

// فایل C مدل که توسط onnx2c تولید شده، در زمان کامپایل مشخص می‌شود
#if defined(MODEL_HEADER)
    #include MODEL_HEADER
#else
    #error "MODEL_HEADER not defined. Please compile using the Makefile."
#endif

#define NUM_IMAGES_TO_TEST 3
#define IMAGE_CHANNELS 3
#define IMAGE_HEIGHT 32
#define IMAGE_WIDTH 32
#define INPUT_SIZE (IMAGE_CHANNELS * IMAGE_HEIGHT * IMAGE_WIDTH)
#define NUM_CLASSES 10

// تابعی برای چاپ کردن آرایه logit
void print_logits(float predictions[1][NUM_CLASSES]) {
    printf("  Logits: [");
    for (int i = 0; i < NUM_CLASSES; ++i) {
        printf("%+9.6f", predictions[0][i]);
        if (i < NUM_CLASSES - 1) {
            printf(", ");
        }
    }
    printf("]\n");
}

// تابعی برای پیدا کردن کلاس پیش‌بینی شده
int get_predicted_class(float predictions[1][NUM_CLASSES]) {
    int predicted_class = -1;
    float max_prob = -1e9; // مقدار اولیه بسیار کوچک
    for (int i = 0; i < NUM_CLASSES; ++i) {
        if (predictions[0][i] > max_prob) {
            max_prob = predictions[0][i];
            predicted_class = i;
        }
    }
    return predicted_class;
}

int main() {
    FILE *image_file, *label_file;
    float (*input_data)[IMAGE_CHANNELS][IMAGE_HEIGHT][IMAGE_WIDTH];
    float (*predictions)[NUM_CLASSES];
    int32_t true_label;

    // تخصیص حافظه
    input_data = malloc(INPUT_SIZE * sizeof(float));
    predictions = malloc(NUM_CLASSES * sizeof(float));
    if (!input_data || !predictions) {
        perror("Failed to allocate memory");
        return 1;
    }

    // باز کردن فایل‌های داده
    image_file = fopen("cifar10_test_images.bin", "rb");
    label_file = fopen("cifar10_test_labels.bin", "rb");
    if (!image_file || !label_file) {
        perror("Error opening data files");
        free(input_data);
        free(predictions);
        if(image_file) fclose(image_file);
        if(label_file) fclose(label_file);
        return 1;
    }

    printf("--- Testing Model: %s ---\n", MODEL_NAME);
    double total_time_spent = 0.0;

    // پردازش سه تصویر اول
    for (int i = 0; i < NUM_IMAGES_TO_TEST; ++i) {
        // خواندن داده‌های تصویر و لیبل
        if (fread(input_data, sizeof(float), INPUT_SIZE, image_file) != INPUT_SIZE ||
            fread(&true_label, sizeof(int32_t), 1, label_file) != 1) {
            fprintf(stderr, "Error reading image or label at index %d\n", i);
            break;
        }

        // --- اندازه‌گیری زمان ---
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        entry((const float (*)[3][32][32])input_data, (float (*)[10])predictions);

        clock_gettime(CLOCK_MONOTONIC, &end);
        total_time_spent += (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        // --- پایان اندازه‌گیری ---

        int predicted_class = get_predicted_class((float (*)[10])predictions);

        printf("\n--- Results for Image %d ---\n", i);
        print_logits((float (*)[10])predictions);
        printf("  Predicted Class: %d, True Label: %d\n", predicted_class, true_label);
    }

    printf("\n--------------------------------------------------\n");
    printf("Total inference time for %d images: %.6f seconds\n", NUM_IMAGES_TO_TEST, total_time_spent);
    printf("--------------------------------------------------\n");

    // پاک‌سازی
    fclose(image_file);
    fclose(label_file);
    free(input_data);
    free(predictions);

    return 0;
}
