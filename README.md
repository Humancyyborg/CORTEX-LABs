


# Aurevia 

**AI-Powered Seizure Prediction Glasses**

Aurevia is a wearable EEG device embedded in glasses that uses machine learning to predict epileptic seizures before they occur, providing critical warning time for users to find safety.

---

##  Overview

Aurevia combines minimally-invasive EEG monitoring with real-time AI prediction to help epilepsy patients anticipate seizures. The system uses 2-4 electrodes positioned near the temples and nose bridge to detect pre-seizure brain activity patterns.

### Key Features

-  **Continuous EEG monitoring** via comfortable eyewear form factor
-  **AI-powered prediction** with CNN-LSTM models
- **Real-time processing** on ESP32 microcontroller or smartphone
-  **Mobile app integration** with emergency alerts
-  **All-day battery life** with USB-C charging
-  **Dual processing modes**: on-device (lightweight) or phone-based (advanced)

---

##  System Architecture

### Hardware Components

- **Electrodes**: Skin-contact sensors on frame (temples/nose bridge)
- **Front-End Amplifier**: Cleans and amplifies weak EEG signals (~μV)
- **ESP32 Microcontroller**: Signal processing, ML inference, wireless transmission
- **USB-C Port**: Charging, data transfer, firmware updates
- **Power Management**: Rechargeable Li-ion battery

### Software Pipeline

EEG Acquisition → Preprocessing → Feature Extraction → ML Model → Alert System

1. **Signal Acquisition** (≥250 Hz sampling)
2. **Preprocessing** (filtering, artifact removal, normalization)
3. **Feature Extraction** (time/frequency domain features)
4. **Prediction** (CNN-LSTM or lightweight models)
5. **Alert Delivery** (haptic, app notification, emergency call)

---

## 🧬 Seizure Prediction Model

### Classification Labels

- **Inter-ictal**: Normal brain activity
- **Pre-ictal**: 0-30 minutes before seizure onset (warning window)
- **Ictal**: During seizure event

### Model Variants

#### Heavy Model (Phone/Cloud)
- **Architecture**: CNN-LSTM / Transformer
- **Input**: 2-4 channels × 2s windows
- **Features**: Spectrogram or raw time series
- **Target**: >80% sensitivity, <0.1 false alarms/hour

#### Tiny Model (ESP32)
- **Architecture**: Lightweight CNN + MLP
- **Size**: <150 KB (quantized)
- **Latency**: <100ms inference time
- **Use case**: Offline operation without phone

---

## 📊 Training Data

### Public Datasets (Phase 1)
- [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/)
- [TUH EEG Corpus](https://isip.piconepress.com/projects/tuh_eeg/)

### Aurevia Pilot Data (Phase 2)
- Real-world recordings from 2-4 electrode glasses layout
- Used for transfer learning and model fine-tuning


---

### Development Roadmap
#### Phase 1: Foundation 
 - Public dataset preprocessing
 - Heavy model architecture
 - Training pipeline with MLflow
 - Basic evaluation metrics

#### Phase 2: Optimization
  - Tiny model quantization
  - ESP32 firmware integration
  - Mobile app prototype
  - Real-time streaming pipeline

### Phase 3: Pilot Testing
  - Collect Aurevia glasses data
  - Transfer learning on 2-4 electrode layout
  - Clinical validation study
  - User experience testing

#### Phase 4: Production 
 - FDA/CE medical device certification
 - Cloud backend infrastructure
 - Production hardware design
 - Commercial launch
