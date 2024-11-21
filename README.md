# **Real-Time Driver Drowsiness Detection System**

---

## **Overview and Results**

This project implements a real-time driver drowsiness detection system that uses a CNN model to classify eye states (open or closed) and combines it with MTCNN for robust face and eye detection. By monitoring the driver’s eye gestures, the system can effectively detect drowsiness and issue alerts to enhance road safety.  

### **Key Objectives:**
- Detect faces and eyes in real-time using **MTCNN**.  
- Classify eye states (open/closed) using a custom-trained **CNN model**.  
- Identify drowsiness based on prolonged eye closure.

### **Results:**
- **Accuracy**: Achieved a **99% accuracy** on the eye state classification task using the CNN model.  
- **Processing Speed**: Average processing time is **18ms/step** for classification.  
- **Memory Efficiency**: Optimized for deployment with moderate computational resources.  

This system represents a significant step towards increasing road safety by identifying drowsy drivers in real-time.

---

## **Source Code**

### **Project Structure**
```
├── data/                # Training and testing datasets
├── models/              # Trained models (CNN, MobileNet-based)
├── src/
│   ├── detection.py     # Face and eye detection using MTCNN
│   ├── classification.py# Eye state classification using CNN
│   ├── drowsiness.py    # Main script for combining detection and classification
│   └── utils.py         # Utility functions for preprocessing and metrics
├── results/
│   ├── confusion_matrix.png
│   ├── training_plot.png
│   └── performance_metrics.csv
├── README.md            # Project documentation
└── requirements.txt     # Python dependencies
```

### **Setup Instructions**
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/drowsiness-detection.git
   cd drowsiness-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the detection system:
   ```bash
   python src/drowsiness.py
   ```

---

## **Performance Metrics**

### **1. Accuracy**
- The CNN model achieved a **99% accuracy** for the binary eye state classification task.  

### **2. Speed**
- **18ms/step** on average for eye state classification with preprocessed frames.

### **3. Memory**
- Model optimized for deployment with moderate GPU resources.

### **Visualizations**
1. **Confusion Matrix**:
   ![Confusion Matrix](results/confusion_matrix.png)

2. **Training Performance**:
   ![Training Plot](results/training_plot.png)

---

## **Installation and Usage**

### **Prerequisites**
- Python >= 3.7
- A compatible GPU (optional for faster inference).

### **Instructions**
1. Clone the repository and install dependencies (see **Setup Instructions**).  
2. To process a video file for drowsiness detection:
   ```bash
   python src/drowsiness.py --input path_to_video.mp4
   ```
3. To run real-time detection using a webcam:
   ```bash
   python src/drowsiness.py --realtime
   ```

---

## **References and Documentation**

### **Datasets Used**:
1. **Eye Classification Dataset**:
   - Used to train the CNN model for eye state classification.  
2. **Video Dataset**:
   - Evaluated the combined performance of the CNN and MTCNN system.  

### **Key Papers**:
1. *Paper on Eye Classification Dataset* ([To be edited]).  
2. *Paper on Video Dataset for Drowsiness Detection* ([To be edited]).  

### **Techniques**:
- **MTCNN**: Multi-task Cascaded Convolutional Networks for real-time face and eye detection.  
- **Custom CNN Model**: Efficient binary classification of eye states.  

---

## **Issues and Contributions**

### **Known Issues**:
1. **Low FPS** in real-time detection:
   - **Cause**: MTCNN is computationally intensive.  
   - **Solution**: Implementing **fastMTCNN** for faster face detection.  
2. **Passenger Detection**:
   - The model might process faces of passengers in the frame.  
   - **Solution**: Revise the system to focus solely on the driver’s face.  
3. **Limited Features**:
   - Currently, drowsiness is based only on eye gestures.  
   - **Solution**: Add yawning detection for a more comprehensive system.

### **How to Contribute**:
- Report issues or suggest features by opening a GitHub issue.  
- Contribute code via pull requests following the guidelines in `CONTRIBUTING.md`.  

---

## **Future Work**

1. **Faster Processing**:  
   - Investigate lightweight models like **fastMTCNN** or quantized MobileNet for deployment on low-end devices.  

2. **Expand Features**:  
   - Integrate yawning detection to complement eye gesture-based analysis.  
   - Focus on filtering out non-driver faces in multi-person frames.  

3. **Driver Behavior Analysis**:  
   - Develop advanced models that combine eye gestures, yawning, and head pose to identify complex driver states.

---

### **Thank You for Your Support!**  
We welcome contributions and feedback to improve this project. Let's make roads safer together!

--- 

You can now customize the placeholders (e.g., dataset/paper names and paths) and add additional information if needed!
