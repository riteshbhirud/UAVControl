
#include <stdio.h>
#include "driver/i2c.h"
#include "mpu6050.h"
#include "esp_log.h"

#define I2C_MASTER_SCL_IO 20      /*!< gpio number for I2C master clock */
#define I2C_MASTER_SDA_IO 17      /*!< gpio number for I2C master data  */
#define I2C_MASTER_NUM I2C_NUM_0  /*!< I2C port number for master dev */
#define I2C_MASTER_FREQ_HZ 400000 /*!< I2C master clock frequency */

#define SAMPLE_FREQUENCY 100
#define SAMPLE_DELAY_MS (1000 / SAMPLE_FREQUENCY)

static const char *TAG = "assignment-1";
static mpu6050_handle_t mpu6050 = NULL;

/**
 * @brief i2c master initialization
 */
static void i2c_bus_init(void)
{
    i2c_config_t conf;
    conf.mode = I2C_MODE_MASTER;
    conf.sda_io_num = (gpio_num_t)I2C_MASTER_SDA_IO;
    conf.sda_pullup_en = GPIO_PULLUP_ENABLE;
    conf.scl_io_num = (gpio_num_t)I2C_MASTER_SCL_IO;
    conf.scl_pullup_en = GPIO_PULLUP_ENABLE;
    conf.master.clk_speed = I2C_MASTER_FREQ_HZ;
    conf.clk_flags = I2C_SCLK_SRC_FLAG_FOR_NOMAL;

    esp_err_t ret = i2c_param_config(I2C_MASTER_NUM, &conf);
    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG, "I2C config returned error");
        return;
    }

    ret = i2c_driver_install(I2C_MASTER_NUM, conf.mode, 0, 0, 0);
    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG, "I2C install returned error");
        return;
    }
}

/**
 * @brief i2c master initialization
 */
static void i2c_sensor_mpu6050_init(void)
{
    esp_err_t ret;

    i2c_bus_init();
    mpu6050 = mpu6050_create(I2C_MASTER_NUM, MPU6050_I2C_ADDRESS);
    if (mpu6050 == NULL)
    {
        ESP_LOGE(TAG, "MPU6050 create returned NULL");
        return;
    }

    ret = mpu6050_config(mpu6050, ACCE_FS_4G, GYRO_FS_500DPS);
    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG, "MPU6050 config error");
        return;
    }

    ret = mpu6050_wake_up(mpu6050);
    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG, "MPU6050 wake up error");
        return;
    }
}

// ______________________________________________________

/**
 * @brief Task to periodically read sensor data
 */
void output_data(void *arg)
{
    mpu6050_acce_value_t acce;
    mpu6050_gyro_value_t gyro;
    esp_err_t ret;

    while (1)
    {
        ret = mpu6050_get_acce(mpu6050, &acce);
        if (ret != ESP_OK)
        {
            ESP_LOGE(TAG, "Failed to get accelerometer data");
        }
        ret = mpu6050_get_gyro(mpu6050, &gyro);
        if (ret != ESP_OK)
        {
            ESP_LOGE(TAG, "Failed to get accelerometer data");
        }

        printf("%.2f, %.2f, %.2f, %.2f, %.2f, %.2f\n",  acce.acce_x, acce.acce_y, acce.acce_z, gyro.gyro_x, gyro.gyro_y, gyro.gyro_z);

        vTaskDelay(pdMS_TO_TICKS(SAMPLE_DELAY_MS));
    }
}

void app_main()
{
    i2c_sensor_mpu6050_init();
    xTaskCreate(output_data, "output_data", 8192, NULL, 5, NULL);
}
