/*
  ESP32-CAM + Teachable Machine 手势识别
  基于 TensorFlowLite_ESP32 的 hello_world 示例修改
  支持串口发送手势指令 (0/1/2)
  支持显示识别耗时
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

// ========== 新增：手势标签到指令的映射 ==========
// 请根据你的 Teachable Machine 实际标签名称修改
struct GestureMapping {
  const char* label;
  uint8_t cmd;
};

// 修改这里的标签名称，使其与你的模型输出一致
GestureMapping gestureMappings[] = {
  {"Fisthand", 0},      // 握拳 → 坐下
  {"Openhand", 1},      // 手掌 → 站立
  {"Vhand", 2},     // V形 → 跳跃
};

const int gestureCount = sizeof(gestureMappings) / sizeof(gestureMappings[0]);

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

// ========== TF Lite 全局变量 ==========
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// 内存池大小
constexpr int kTensorArenaSize = 90 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// ========== 新增：根据标签查找对应的指令 ==========
int findGestureCommand(const char* label) {
  for (int i = 0; i < gestureCount; i++) {
    if (strcmp(label, gestureMappings[i].label) == 0) {
      return gestureMappings[i].cmd;
    }
  }
  return -1;  // 未找到
}

// ========== 新增：发送手势指令 ==========
void sendGestureCommand(uint8_t cmd) {
  // 发送原始字节指令
  Serial.write(cmd);
  
  // 打印发送内容（方便调试）
  Serial.print(" [发送指令: ");
  Serial.print(cmd);
  switch(cmd) {
    case 0: Serial.print(" - 坐下"); break;
    case 1: Serial.print(" - 站立"); break;
    case 2: Serial.print(" - 跳跃"); break;
    default: break;
  }
  Serial.println("]");
}

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
  while (!Serial);
  Serial.println("ESP32-CAM 手势识别启动");
  Serial.println("串口已初始化，波特率: 115200");

  // 1. 摄像头
  init_camera();

  // 2. TF Lite 初始化
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(g_person_detect_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("模型版本不匹配");
    return;
  }

  // 注册模型需要的操作
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
  // ========== 开始计时 ==========
  unsigned long startTime = millis();
  
  // 拍照
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("拍照失败");
    delay(1000);
    return;
  }
  
  // 记录拍照完成时间
  unsigned long captureTime = millis();

  // 灰度 + 归一化到 int8（-128 ~ 127）
  for (int i = 0; i < kNumCols * kNumRows; i++) {
    input->data.int8[i] = (int8_t)(fb->buf[i] - 128);
  }
  esp_camera_fb_return(fb);
  
  // 记录预处理完成时间
  unsigned long preprocessTime = millis();

  // 推理
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("推理失败");
    return;
  }
  
  // 记录推理完成时间
  unsigned long inferenceTime = millis();

  // 解析输出（kCategoryCount 个类别）
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
    const char* detected_gesture = kCategoryLabels[max_index];
    Serial.printf(">>> 手势: %s\n", detected_gesture);
    
    // 查找指令并发送
    int cmd = findGestureCommand(detected_gesture);
    if (cmd != -1) {
      sendGestureCommand((uint8_t)cmd);
    } else {
      Serial.println(">>> 未匹配到指令（该手势未映射）");
    }
  } else {
    Serial.println(">>> 未识别");
  }
  
  // 记录发送完成时间
  unsigned long sendTime = millis();
  
  // ========== 输出时间统计 ==========
  Serial.println("----------------------------------------");
  Serial.println("【识别耗时统计】");
  Serial.printf("拍照耗时: %lu ms\n", captureTime - startTime);
  Serial.printf("预处理耗时: %lu ms\n", preprocessTime - captureTime);
  Serial.printf("推理耗时: %lu ms\n", inferenceTime - preprocessTime);
  Serial.printf("发送耗时: %lu ms\n", sendTime - inferenceTime);
  Serial.printf("总耗时: %lu ms\n", sendTime - startTime);
  Serial.println("========================================");

  delay(2000);
}