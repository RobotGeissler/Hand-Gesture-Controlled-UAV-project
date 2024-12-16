#include <stdio.h>
#include "unity.h"
#include "driver/i2c.h"
#include "driver/uart.h"
#include "mpu6050.h"
#include "esp_system.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_vfs.h"
#include "esp_vfs_fat.h"    
#include "esp_spiffs.h"

#define I2C_MASTER_SCL_IO 0      /*!< GPIO number for I2C master clock for MPU6050 1 */
#define I2C_MASTER_SDA_IO 1      /*!< GPIO number for I2C master data for MPU6050 1 */
#define I2C_MASTER_SCL_IO_2 2    /*!< GPIO number for I2C master clock for MPU6050 2 */
#define I2C_MASTER_SDA_IO_2 42   /*!< GPIO number for I2C master data for MPU6050 2 */
#define I2C_MASTER_NUM I2C_NUM_0 /*!< I2C port number for MPU6050 1 */
#define I2C_MASTER_NUM_2 I2C_NUM_1 /*!< I2C port number for MPU6050 2 */
#define I2C_MASTER_FREQ_HZ 10000 /*!< I2C master clock frequency */

static const char *TAG = "mpu6050 test";
static mpu6050_handle_t mpu6050_1 = NULL;
static mpu6050_handle_t mpu6050_2 = NULL;

static void i2c_bus_init(gpio_num_t scl_io, gpio_num_t sda_io, i2c_port_t port)
{
    i2c_config_t conf;
    conf.mode = I2C_MODE_MASTER;
    conf.sda_io_num = sda_io;
    conf.sda_pullup_en = GPIO_PULLUP_ENABLE;
    conf.scl_io_num = scl_io;
    conf.scl_pullup_en = GPIO_PULLUP_ENABLE;
    conf.master.clk_speed = I2C_MASTER_FREQ_HZ;
    conf.clk_flags = I2C_SCLK_SRC_FLAG_FOR_NOMAL;

    esp_err_t ret = i2c_param_config(port, &conf);
    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG, "I2C config returned error for port %d", port);
        return;
    }

    ret = i2c_driver_install(port, conf.mode, 0, 0, 0);
    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG, "I2C install returned error for port %d", port);
        return;
    }
}

static void i2c_sensor_mpu6050_init(void)
{
    esp_err_t ret;

    i2c_bus_init(I2C_MASTER_SCL_IO, I2C_MASTER_SDA_IO, I2C_MASTER_NUM);
    mpu6050_1 = mpu6050_create(I2C_MASTER_NUM, MPU6050_I2C_ADDRESS);
    if (mpu6050_1 == NULL)
    {
        ESP_LOGE(TAG, "MPU6050 create returned NULL for MPU1");
        return;
    }

    ret = mpu6050_config(mpu6050_1, ACCE_FS_4G, GYRO_FS_500DPS);
    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG, "MPU6050 config error for MPU1");
        return;
    }

    ret = mpu6050_wake_up(mpu6050_1);
    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG, "MPU6050 wake up error for MPU1");
        return;
    }

    i2c_bus_init(I2C_MASTER_SCL_IO_2, I2C_MASTER_SDA_IO_2, I2C_MASTER_NUM_2);
    mpu6050_2 = mpu6050_create(I2C_MASTER_NUM_2, MPU6050_I2C_ADDRESS);
    if (mpu6050_2 == NULL)
    {
        ESP_LOGE(TAG, "MPU6050 create returned NULL for MPU2");
        return;
    }

    ret = mpu6050_config(mpu6050_2, ACCE_FS_4G, GYRO_FS_500DPS);
    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG, "MPU6050 config error for MPU2");
        return;
    }

    ret = mpu6050_wake_up(mpu6050_2);
    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG, "MPU6050 wake up error for MPU2");
        return;
    }
}

void app_main()
{
    esp_err_t ret;
    mpu6050_acce_value_t acce1, acce2;
    mpu6050_gyro_value_t gyro1, gyro2;

    // ret = init_spiffs();
    // if (ret != ESP_OK)
    // {
    //     ESP_LOGE(TAG, "Failed to initialize SPIFFS");
    // }

    i2c_sensor_mpu6050_init();

    vTaskDelay(3000 / portTICK_PERIOD_MS);

    int64_t time = esp_timer_get_time();
    while (esp_timer_get_time() - time < 7 * 1000000)
    {
        ret = mpu6050_get_acce(mpu6050_1, &acce1);
        if (ret != ESP_OK)
        {
            ESP_LOGE(TAG, "Failed to get accelerometer data for MPU1");
        }
        // ESP_LOGI(TAG, "MPU1 ", acce1.acce_x, acce1.acce_y, acce1.acce_z);

        ret = mpu6050_get_gyro(mpu6050_1, &gyro1);
        if (ret != ESP_OK)
        {
            ESP_LOGE(TAG, "Failed to get gyroscope data for MPU1");
        }
        ESP_LOGI(TAG, "MPU1 acce_x:%.5f, acce_y:%.5f, acce_z:%.5f, gyro_x:%.5f, gyro_y:%.5f, gyro_z:%.5f", acce1.acce_x, acce1.acce_y, acce1.acce_z, gyro1.gyro_x, gyro1.gyro_y, gyro1.gyro_z);

        ret = mpu6050_get_acce(mpu6050_2, &acce2);
        if (ret != ESP_OK)
        {
            ESP_LOGE(TAG, "Failed to get accelerometer data for MPU2");
        }
        // ESP_LOGI(TAG, "MPU2 acce_x:%.5f, acce_y:%.5f, acce_z:%.5f", acce2.acce_x, acce2.acce_y, acce2.acce_z);

        ret = mpu6050_get_gyro(mpu6050_2, &gyro2);
        if (ret != ESP_OK)
        {
            ESP_LOGE(TAG, "Failed to get gyroscope data for MPU2");
        }
        // ESP_LOGI(TAG, "MPU2 gyro_x:%.5f, gyro_y:%.5f, gyro_z:%.5f", gyro2.gyro_x, gyro2.gyro_y, gyro2.gyro_z);
        ESP_LOGI(TAG, "MPU2 acce_x:%.5f, acce_y:%.5f, acce_z:%.5f, gyro_x:%.5f, gyro_y:%.5f, gyro_z:%.5f", acce2.acce_x, acce2.acce_y, acce2.acce_z, gyro2.gyro_x, gyro2.gyro_y, gyro2.gyro_z);
    }

    mpu6050_delete(mpu6050_1);
    mpu6050_delete(mpu6050_2);

    ret = i2c_driver_delete(I2C_MASTER_NUM);
    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG, "Failed to delete I2C driver for MPU1");
    }

    ret = i2c_driver_delete(I2C_MASTER_NUM_2);
    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG, "Failed to delete I2C driver for MPU2");
    }

    esp_vfs_spiffs_unregister(NULL);
}
