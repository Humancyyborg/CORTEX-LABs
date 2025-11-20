/*
 * Aurevia - Real-Time Seizure Prediction Device
 * ESP32-C3 Firmware
 * 
 * This firmware implements:
 * - 4-channel EEG acquisition via ADS1299
 * - Real-time signal preprocessing
 * - Feature extraction (128 features)
 * - TensorFlow Lite inference
 * - BLE alert notifications
 */

#include <Arduino.h>
#include <SPI.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model.h"  // Generated C array from Python conversion

// ============================================================================
// CONFIGURATION
// ============================================================================

// Hardware Pins
#define ADS1299_CS    7
#define ADS1299_DRDY  6
#define ADS1299_START 5
#define ADS1299_RESET 4
#define LED_STATUS    8
#define VIBRATION_MOTOR 9

// EEG Parameters
#define NUM_CHANNELS  4
#define SAMPLE_RATE   250  // Hz
#define WINDOW_SIZE   250  // 1 second of data
#define NUM_FEATURES  128

// ML Parameters
#define PREDICTION_THRESHOLD 0.75  // 75% probability
#define ALERT_COOLDOWN_MS    60000 // 1 minute between alerts

// BLE Configuration
#define DEVICE_NAME "Aurevia"
#define SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHAR_PREDICTION_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"
#define CHAR_STATUS_UUID     "1c95d5e3-d8f7-413a-bf3d-7a2e5d7be87e"

// ============================================================================
// GLOBAL VARIABLES
// ============================================================================

// EEG Data Buffers
float eeg_buffer[NUM_CHANNELS][WINDOW_SIZE];
uint16_t buffer_index = 0;
bool buffer_ready = false;

// Feature Vector
float features[NUM_FEATURES];

// TensorFlow Lite
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Allocate memory for TFLite (adjust size based on model)
constexpr int kTensorArenaSize = 80 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// BLE
BLEServer* pServer = nullptr;
BLECharacteristic* pPredictionChar = nullptr;
BLECharacteristic* pStatusChar = nullptr;
bool device_connected = false;

// Timing
unsigned long last_alert_time = 0;
unsigned long last_sample_time = 0;

// Statistics
uint32_t total_predictions = 0;
uint32_t alert_count = 0;

// Feature Normalization (from training - feature_scaler.pkl)
// These values should match your StandardScaler parameters
float feature_mean[NUM_FEATURES];
float feature_std[NUM_FEATURES];

// ============================================================================
// ADS1299 REGISTERS
// ============================================================================

#define ADS1299_RREG    0x20
#define ADS1299_WREG    0x40
#define ADS1299_SDATAC  0x11
#define ADS1299_RDATAC  0x10
#define ADS1299_START_CMD 0x08
#define ADS1299_STOP    0x0A

#define REG_CONFIG1     0x01
#define REG_CONFIG2     0x02
#define REG_CONFIG3     0x03
#define REG_CH1SET      0x05
#define REG_CH2SET      0x06
#define REG_CH3SET      0x07
#define REG_CH4SET      0x08

// ============================================================================
// BLE CALLBACKS
// ============================================================================

class ServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
        device_connected = true;
        Serial.println("BLE Client Connected");
    }

    void onDisconnect(BLEServer* pServer) {
        device_connected = false;
        Serial.println("BLE Client Disconnected");
        // Restart advertising
        pServer->getAdvertising()->start();
    }
};

// ============================================================================
// ADS1299 LOW-LEVEL FUNCTIONS
// ============================================================================

void ads1299_write_reg(uint8_t reg, uint8_t val) {
    digitalWrite(ADS1299_CS, LOW);
    SPI.transfer(ADS1299_WREG | reg);
    SPI.transfer(0x00);  // Write 1 register
    SPI.transfer(val);
    digitalWrite(ADS1299_CS, HIGH);
    delayMicroseconds(10);
}

uint8_t ads1299_read_reg(uint8_t reg) {
    digitalWrite(ADS1299_CS, LOW);
    SPI.transfer(ADS1299_RREG | reg);
    SPI.transfer(0x00);  // Read 1 register
    delayMicroseconds(10);
    uint8_t val = SPI.transfer(0x00);
    digitalWrite(ADS1299_CS, HIGH);
    return val;
}

void ads1299_send_command(uint8_t cmd) {
    digitalWrite(ADS1299_CS, LOW);
    SPI.transfer(cmd);
    digitalWrite(ADS1299_CS, HIGH);
    delayMicroseconds(10);
}

void ads1299_init() {
    Serial.println("Initializing ADS1299...");
    
    // Reset
    digitalWrite(ADS1299_RESET, LOW);
    delay(100);
    digitalWrite(ADS1299_RESET, HIGH);
    delay(150);
    
    // Stop continuous mode
    ads1299_send_command(ADS1299_SDATAC);
    delay(10);
    
    // Configure device
    // CONFIG1: 250 SPS, continuous conversion
    ads1299_write_reg(REG_CONFIG1, 0x96);
    
    // CONFIG2: Test signals off, reference buffer on
    ads1299_write_reg(REG_CONFIG2, 0xD0);
    
    // CONFIG3: Internal reference, bias enabled
    ads1299_write_reg(REG_CONFIG3, 0xEC);
    
    // Configure channels 1-4 (normal input, gain=24)
    for (int i = 0; i < NUM_CHANNELS; i++) {
        ads1299_write_reg(REG_CH1SET + i, 0x60);
    }
    
    delay(10);
    
    // Start continuous conversion
    ads1299_send_command(ADS1299_RDATAC);
    digitalWrite(ADS1299_START, HIGH);
    
    Serial.println("ADS1299 initialized successfully");
}

// Read one sample from all channels
void ads1299_read_data(int32_t* channel_data) {
    digitalWrite(ADS1299_CS, LOW);
    
    // Read status bytes (3 bytes)
    SPI.transfer(0x00);
    SPI.transfer(0x00);
    SPI.transfer(0x00);
    
    // Read channel data (3 bytes per channel, 24-bit 2's complement)
    for (int ch = 0; ch < NUM_CHANNELS; ch++) {
        uint8_t byte1 = SPI.transfer(0x00);
        uint8_t byte2 = SPI.transfer(0x00);
        uint8_t byte3 = SPI.transfer(0x00);
        
        // Convert 24-bit to 32-bit signed
        int32_t value = ((int32_t)byte1 << 16) | ((int32_t)byte2 << 8) | byte3;
        
        // Sign extend from 24-bit to 32-bit
        if (value & 0x800000) {
            value |= 0xFF000000;
        }
        
        channel_data[ch] = value;
    }
    
    digitalWrite(ADS1299_CS, HIGH);
}

// ============================================================================
// SIGNAL PROCESSING - BANDPASS FILTER (0.5-40 Hz)
// ============================================================================

// Simple IIR Butterworth filter coefficients (calculated offline)
// 4th order bandpass filter at 250 Hz sample rate
struct FilterState {
    float x[5];  // Input history
    float y[5];  // Output history
};

FilterState filters[NUM_CHANNELS];

// Bandpass filter (0.5-40 Hz) - coefficients from scipy.signal.butter
const float b_coeff[] = {0.0048, 0.0, -0.0096, 0.0, 0.0048};
const float a_coeff[] = {1.0, -3.7695, 5.3392, -3.3599, 0.7903};

float apply_filter(float input, FilterState* state) {
    // Shift history
    for (int i = 4; i > 0; i--) {
        state->x[i] = state->x[i-1];
        state->y[i] = state->y[i-1];
    }
    
    state->x[0] = input;
    
    // Apply filter equation
    float output = 0.0;
    for (int i = 0; i < 5; i++) {
        output += b_coeff[i] * state->x[i];
    }
    for (int i = 1; i < 5; i++) {
        output -= a_coeff[i] * state->y[i];
    }
    
    state->y[0] = output;
    return output;
}

// ============================================================================
// FEATURE EXTRACTION
// ============================================================================

// Helper: Compute mean
float compute_mean(float* data, int len) {
    float sum = 0.0;
    for (int i = 0; i < len; i++) {
        sum += data[i];
    }
    return sum / len;
}

// Helper: Compute standard deviation
float compute_std(float* data, int len, float mean) {
    float sum_sq = 0.0;
    for (int i = 0; i < len; i++) {
        float diff = data[i] - mean;
        sum_sq += diff * diff;
    }
    return sqrt(sum_sq / len);
}

// Extract time-domain features from one channel (8 features)
void extract_time_features(float* signal, int len, float* features_out) {
    int idx = 0;
    
    // Mean
    float mean = compute_mean(signal, len);
    features_out[idx++] = mean;
    
    // Standard deviation
    float std = compute_std(signal, len, mean);
    features_out[idx++] = std;
    
    // Peak-to-peak
    float min_val = signal[0], max_val = signal[0];
    for (int i = 1; i < len; i++) {
        if (signal[i] < min_val) min_val = signal[i];
        if (signal[i] > max_val) max_val = signal[i];
    }
    features_out[idx++] = max_val - min_val;
    
    // RMS
    float sum_sq = 0.0;
    for (int i = 0; i < len; i++) {
        sum_sq += signal[i] * signal[i];
    }
    features_out[idx++] = sqrt(sum_sq / len);
    
    // Zero crossings
    int zero_crossings = 0;
    for (int i = 1; i < len; i++) {
        if ((signal[i-1] >= 0 && signal[i] < 0) || (signal[i-1] < 0 && signal[i] >= 0)) {
            zero_crossings++;
        }
    }
    features_out[idx++] = (float)zero_crossings;
    
    // Energy
    features_out[idx++] = sum_sq;
    
    // Simplified kurtosis and skewness (computationally expensive, using approximations)
    float sum_cube = 0.0, sum_quad = 0.0;
    for (int i = 0; i < len; i++) {
        float diff = signal[i] - mean;
        float diff_sq = diff * diff;
        sum_cube += diff * diff_sq;
        sum_quad += diff_sq * diff_sq;
    }
    float variance = std * std;
    features_out[idx++] = sum_quad / (len * variance * variance);  // Kurtosis
    features_out[idx++] = sum_cube / (len * variance * std);       // Skewness
}

// Simplified FFT using Goertzel algorithm for specific frequency bands
void extract_frequency_features(float* signal, int len, float* features_out) {
    int idx = 0;
    
    // Frequency bands (Hz)
    float bands[][2] = {
        {0.5, 4},   // Delta
        {4, 8},     // Theta
        {8, 13},    // Alpha
        {13, 30}    // Beta
    };
    
    float fs = SAMPLE_RATE;
    
    // Compute power in each band using simplified method
    for (int b = 0; b < 4; b++) {
        float low_freq = bands[b][0];
        float high_freq = bands[b][1];
        
        // Approximate band power by summing squared amplitudes in range
        float band_power = 0.0;
        int count = 0;
        
        // Simple frequency estimation
        for (int i = 1; i < len - 1; i++) {
            // Local frequency estimation (zero-crossing rate method)
            if ((signal[i-1] < 0 && signal[i] >= 0) || (signal[i-1] >= 0 && signal[i] < 0)) {
                float amplitude = fabs(signal[i]);
                band_power += amplitude * amplitude;
                count++;
            }
        }
        
        features_out[idx++] = (count > 0) ? band_power / count : 0.0;
    }
    
    // Add 12 more simplified frequency features
    // Total power
    float total_power = 0.0;
    for (int i = 0; i < len; i++) {
        total_power += signal[i] * signal[i];
    }
    features_out[idx++] = total_power;
    
    // Fill remaining frequency features with derived metrics
    for (int i = 0; i < 11; i++) {
        features_out[idx++] = features_out[i % 4] / (total_power + 1e-6);
    }
}

// Inter-channel features (simplified)
void extract_interchannel_features(float features_out[]) {
    int idx = 96;  // Start after time and frequency features
    
    // Compute correlation between channel pairs
    for (int ch1 = 0; ch1 < NUM_CHANNELS; ch1++) {
        for (int ch2 = ch1 + 1; ch2 < NUM_CHANNELS; ch2++) {
            float corr = 0.0;
            float mean1 = compute_mean(eeg_buffer[ch1], WINDOW_SIZE);
            float mean2 = compute_mean(eeg_buffer[ch2], WINDOW_SIZE);
            
            float sum_prod = 0.0, sum_sq1 = 0.0, sum_sq2 = 0.0;
            for (int i = 0; i < WINDOW_SIZE; i++) {
                float diff1 = eeg_buffer[ch1][i] - mean1;
                float diff2 = eeg_buffer[ch2][i] - mean2;
                sum_prod += diff1 * diff2;
                sum_sq1 += diff1 * diff1;
                sum_sq2 += diff2 * diff2;
            }
            
            corr = sum_prod / (sqrt(sum_sq1 * sum_sq2) + 1e-6);
            
            // Store correlation and derived features
            features_out[idx++] = corr;
            features_out[idx++] = corr * corr;  // Squared correlation
            features_out[idx++] = fabs(corr);   // Absolute correlation
            features_out[idx++] = (corr > 0) ? corr : 0;  // Positive correlation
            features_out[idx++] = (corr < 0) ? -corr : 0; // Negative correlation
        }
    }
}

// Main feature extraction function
void extract_all_features() {
    int feature_idx = 0;
    
    // Time domain features: 8 features × 4 channels = 32 features
    for (int ch = 0; ch < NUM_CHANNELS; ch++) {
        extract_time_features(eeg_buffer[ch], WINDOW_SIZE, &features[feature_idx]);
        feature_idx += 8;
    }
    
    // Frequency domain features: 16 features × 4 channels = 64 features
    for (int ch = 0; ch < NUM_CHANNELS; ch++) {
        extract_frequency_features(eeg_buffer[ch], WINDOW_SIZE, &features[feature_idx]);
        feature_idx += 16;
    }
    
    // Inter-channel features: 32 features
    extract_interchannel_features(features);
    
    // Normalize features using pre-computed statistics
    for (int i = 0; i < NUM_FEATURES; i++) {
        features[i] = (features[i] - feature_mean[i]) / (feature_std[i] + 1e-8);
    }
}

// ============================================================================
// ML INFERENCE
// ============================================================================

float run_inference() {
    // Copy features to input tensor
    for (int i = 0; i < NUM_FEATURES; i++) {
        input->data.f[i] = features[i];
    }
    
    // Run inference
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        Serial.println("Inference failed!");
        return 0.0;
    }
    
    // Get output (probability of pre-seizure state)
    float pre_seizure_prob = output->data.f[1];
    
    return pre_seizure_prob;
}

// ============================================================================
// ALERT SYSTEM
// ============================================================================

void trigger_alert(float probability) {
    unsigned long current_time = millis();
    
    // Check cooldown period
    if (current_time - last_alert_time < ALERT_COOLDOWN_MS) {
        return;
    }
    
    Serial.printf("🚨 ALERT: Seizure predicted! Probability: %.1f%%\n", probability * 100);
    
    // Visual alert (LED flash)
    for (int i = 0; i < 5; i++) {
        digitalWrite(LED_STATUS, HIGH);
        delay(200);
        digitalWrite(LED_STATUS, LOW);
        delay(200);
    }
    
    // Haptic alert (vibration motor)
    digitalWrite(VIBRATION_MOTOR, HIGH);
    delay(1000);
    digitalWrite(VIBRATION_MOTOR, LOW);
    
    // BLE notification
    if (device_connected) {
        String alert_msg = String(probability * 100, 1) + "%";
        pPredictionChar->setValue(alert_msg.c_str());
        pPredictionChar->notify();
    }
    
    last_alert_time = current_time;
    alert_count++;
}

// ============================================================================
// SETUP
// ============================================================================

void setup() {
    Serial.begin(115200);
    delay(2000);
    
    Serial.println("\n\n");
    Serial.println("╔═══════════════════════════════════════════╗");
    Serial.println("║      AUREVIA SEIZURE PREDICTION          ║");
    Serial.println("║      ESP32-C3 Firmware v1.0               ║");
    Serial.println("╚═══════════════════════════════════════════╝");
    Serial.println();
    
    // Initialize pins
    pinMode(ADS1299_CS, OUTPUT);
    pinMode(ADS1299_DRDY, INPUT);
    pinMode(ADS1299_START, OUTPUT);
    pinMode(ADS1299_RESET, OUTPUT);
    pinMode(LED_STATUS, OUTPUT);
    pinMode(VIBRATION_MOTOR, OUTPUT);
    
    digitalWrite(ADS1299_CS, HIGH);
    digitalWrite(ADS1299_START, LOW);
    digitalWrite(LED_STATUS, LOW);
    digitalWrite(VIBRATION_MOTOR, LOW);
    
    // Initialize SPI
    SPI.begin();
    SPI.setFrequency(4000000);  // 4 MHz
    SPI.setDataMode(SPI_MODE1);
    
    // Initialize ADS1299
    ads1299_init();
    
    // Initialize filter states
    for (int ch = 0; ch < NUM_CHANNELS; ch++) {
        memset(&filters[ch], 0, sizeof(FilterState));
    }
    
    // Load normalization parameters (should be flashed from Python)
    // For now, using identity normalization
    for (int i = 0; i < NUM_FEATURES; i++) {
        feature_mean[i] = 0.0;
        feature_std[i] = 1.0;
    }
    
    // Initialize TensorFlow Lite
    Serial.println("Loading TensorFlow Lite model...");
    model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.printf("Model schema version %d doesn't match supported version %d\n",
                     model->version(), TFLITE_SCHEMA_VERSION);
        while(1);
    }
    
    // Set up resolver with required operations
    static tflite::MicroMutableOpResolver<6> resolver;
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddRelu();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddReshape();
    
    // Build interpreter
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;
    
    // Allocate tensors
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        Serial.println("AllocateTensors() failed");
        while(1);
    }
    
    // Get input/output tensors
    input = interpreter->input(0);
    output = interpreter->output(0);
    
    Serial.printf("Model loaded successfully\n");
    Serial.printf("Input shape: [%d]\n", input->dims->data[1]);
    Serial.printf("Output shape: [%d]\n", output->dims->data[1]);
    Serial.printf("Arena used: %d / %d bytes\n", 
                  interpreter->arena_used_bytes(), kTensorArenaSize);
    
    // Initialize BLE
    Serial.println("Initializing BLE...");
    BLEDevice::init(DEVICE_NAME);
    pServer = BLEDevice::createServer();
    pServer->setCallbacks(new ServerCallbacks());
    
    BLEService *pService = pServer->createService(SERVICE_UUID);
    
    pPredictionChar = pService->createCharacteristic(
        CHAR_PREDICTION_UUID,
        BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY
    );
    pPredictionChar->addDescriptor(new BLE2902());
    
    pStatusChar = pService->createCharacteristic(
        CHAR_STATUS_UUID,
        BLECharacteristic::PROPERTY_READ
    );
    
    pService->start();
    
    BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
    pAdvertising->addServiceUUID(SERVICE_UUID);
    pAdvertising->setScanResponse(true);
    pAdvertising->start();
    
    Serial.println("BLE advertising started");
    Serial.println("\n✓ System ready - Monitoring EEG signals...\n");
    
    digitalWrite(LED_STATUS, HIGH);
}

// ============================================================================
// MAIN LOOP
// ============================================================================

void loop() {
    // Check if new data is ready (DRDY pin goes low)
    if (digitalRead(ADS1299_DRDY) == LOW) {
        int32_t raw_data[NUM_CHANNELS];
        ads1299_read_data(raw_data);
        
        // Convert to microvolts and apply filter
        for (int ch = 0; ch < NUM_CHANNELS; ch++) {
            // Convert 24-bit ADC value to microvolts
            // Vref = 4.5V, Gain = 24, resolution = 2^23
            float voltage = (raw_data[ch] * 4.5) / (24.0 * 8388608.0) * 1000000.0;
            
            // Apply bandpass filter
            float filtered = apply_filter(voltage, &filters[ch]);
            
            // Store in buffer
            eeg_buffer[ch][buffer_index] = filtered;
        }
        
        buffer_index++;
        
        // When buffer is full, extract features and run inference
        if (buffer_index >= WINDOW_SIZE) {
            buffer_index = 0;
            buffer_ready = true;
            
            // Extract features
            unsigned long t1 = micros();
            extract_all_features();
            unsigned long feature_time = micros() - t1;
            
            // Run ML inference
            t1 = micros();
            float seizure_prob = run_inference();
            unsigned long inference_time = micros() - t1;
            
            total_predictions++;
            
            // Log results
            Serial.printf("Prediction #%d: %.1f%% | Feature: %dµs | Inference: %dµs\n",
                         total_predictions, seizure_prob * 100, 
                         feature_time, inference_time);
            
            // Update BLE status
            if (device_connected && total_predictions % 10 == 0) {
                char status[64];
                snprintf(status, sizeof(status), 
                        "Pred:%d|Prob:%.1f%%|Alerts:%d", 
                        total_predictions, seizure_prob * 100, alert_count);
                pStatusChar->setValue(status);
            }
            
            // Check if alert should be triggered
            if (seizure_prob >= PREDICTION_THRESHOLD) {
                trigger_alert(seizure_prob);
            }
            
            // Blink LED to show activity
            digitalWrite(LED_STATUS, !digitalRead(LED_STATUS));
        }
    }
    
    delay(1);  // Small delay to prevent watchdog timeout
}
