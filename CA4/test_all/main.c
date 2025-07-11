#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

// فایل C تولید شده توسط onnx2c را include کنید
#include "q8_new3.c" 

#define IMAGE_BATCH 1
#define IMAGE_CHANNELS 3
#define IMAGE_HEIGHT 32
#define IMAGE_WIDTH 32
#define INPUT_SIZE (IMAGE_BATCH * IMAGE_CHANNELS * IMAGE_HEIGHT * IMAGE_WIDTH)

#define NUM_CLASSES 10

int main() {
    FILE *image_file, *label_file;
    float (*input_data)[IMAGE_CHANNELS][IMAGE_HEIGHT][IMAGE_WIDTH] =
        (float (*)[IMAGE_CHANNELS][IMAGE_HEIGHT][IMAGE_WIDTH]) malloc(INPUT_SIZE * sizeof(float));

    float (*predictions)[NUM_CLASSES] =
        (float (*)[NUM_CLASSES]) malloc(IMAGE_BATCH * NUM_CLASSES * sizeof(float));

    int32_t true_label;
    int correct_predictions = 0;
    int total_images = 0;

    if (!input_data || !predictions) {
        perror("Failed to allocate memory");
        return 1;
    }

    // *** FIX: Added fopen for the image file ***
    image_file = fopen("cifar10_test_images.bin", "rb");
    label_file = fopen("cifar10_test_labels.bin", "rb");

    if (!image_file || !label_file) {
        perror("Error opening data files. Make sure 'cifar10_test_images.bin' and 'cifar10_test_labels.bin' exist.");
        free(input_data);
        free(predictions);
        // اگر یکی از فایل‌ها باز نشد، دیگری را (اگر باز شده باشد) ببندید
        if (image_file) fclose(image_file);
        if (label_file) fclose(label_file);
        return 1;
    }

    printf("Starting model testing...\n");

    // حلقه برای خواندن تمام تصاویر در فایل
    while (fread(input_data, sizeof(float), INPUT_SIZE, image_file) == INPUT_SIZE &&
           fread(&true_label, sizeof(int32_t), 1, label_file) == 1) {

        total_images++;

        // فراخوانی تابع اصلی مدل
        entry((const float (*)[3][32][32])input_data, (float (*)[10])predictions);

        // پیدا کردن کلاس پیش‌بینی شده
        int predicted_class = -1;
        float max_prob = -2e30; // مقدار اولیه بسیار کوچک
        for (int i = 0; i < NUM_CLASSES; ++i) {
            if (predictions[0][i] > max_prob) {
                max_prob = predictions[0][i];
                predicted_class = i;
            }
        }

        if (predicted_class == true_label) {
            correct_predictions++;
        }

        // نمایش پیشرفت
        if (total_images % 100 == 0) {
            printf("Processed %d images. Current accuracy: %.2f%%\n",
                   total_images, (float)correct_predictions / total_images * 100.0f);
        }
    }

    printf("\nTesting complete.\n");
    printf("Total images tested: %d\n", total_images);
    printf("Correct predictions: %d\n", correct_predictions);
    
    // جلوگیری از تقسیم بر صفر اگر هیچ تصویری پردازش نشده باشد
    if (total_images > 0) {
        printf("Accuracy: %.2f%%\n", (float)correct_predictions / total_images * 100.0f);
    }

    fclose(image_file);
    fclose(label_file);
    free(input_data);
    free(predictions);

    return 0;
}
