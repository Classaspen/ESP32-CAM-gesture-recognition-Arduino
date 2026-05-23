/*
  ESP32-CAM + Teachable Machine 手势识别
  基于 TensorFlowLite_ESP32 的 hello_world 示例修改
*/

#include <TensorFlowLite_ESP32.h>
#include "esp_camera.h"

// 你的模型文件（Teachable Machine 导出的）
#include "model_settings.h"
#include "person_detect_model_data.h"

// hello_world 示例需要的 TF Lite 头文件
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
// #include "tensorflow/lite/version.h"

// ========== ESP32-CAM 引脚定义 ==========
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// ========== TF Lite 全局变量（hello_world 风格）==========
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// 内存池大小（如果内存不够可以适当调小）
constexpr int kTensorArenaSize = 90 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// ========== 摄像头初始化 ==========
void init_camera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_GRAYSCALE;
  config.frame_size = FRAMESIZE_96X96;
  config.jpeg_quality = 12;
  config.fb_count = 1;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera error: 0x%x\n", err);
    ESP.restart();
  }
  Serial.println("Camera initial successfully");
}

// ========== 主程序 setup ==========
void setup() {
  Serial.begin(115200);
  while (!Serial);  // 等待串口连接（调试用）
  Serial.println("ESP32-CAM 手势识别启动");

  // 1. 摄像头
  init_camera();

  // 2. TF Lite 初始化（完全按照 hello_world 的风格）
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(g_person_detect_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("模型版本不匹配");
    return;
  }

  // 只注册模型真正需要的操作（如果缺某个算子，编译会报错，到时再加）
  static tflite::MicroMutableOpResolver<6> resolver;
  resolver.AddAveragePool2D();
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddReshape();
  resolver.AddSoftmax();
  resolver.AddFullyConnected();

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("内存分配失败");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("推理初始化完成，开始识别...");
}

// ========== 主循环 loop ==========
void loop() {
  // 拍照
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("拍照失败");
    delay(1000);
    return;
  }

  // 灰度 + 归一化到 int8（-128 ~ 127）
  for (int i = 0; i < kNumCols * kNumRows; i++) {
    input->data.int8[i] = (int8_t)(fb->buf[i] - 128);
  }
  esp_camera_fb_return(fb);

  // 推理
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("推理失败");
    return;
  }

  // 解析输出（3 个类别）
  int8_t max_score = -128;
  int max_index = 0;
  Serial.println("=== 识别结果 ===");
  for (int i = 0; i < kCategoryCount; i++) {
    int8_t score = output->data.int8[i];
    float percent = (score + 128) * 100.0f / 255.0f;
    Serial.printf("%s: %.1f%%\n", kCategoryLabels[i], percent);
    if (score > max_score) {
      max_score = score;
      max_index = i;
    }
  }

  if ((max_score + 128) > 128) {  // 置信度 > 50%
    Serial.printf(">>> 手势: %s\n", kCategoryLabels[max_index]);
  } else {
    Serial.println(">>> 未识别");
  }

  delay(2000);
}