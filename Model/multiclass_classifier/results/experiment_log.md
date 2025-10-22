## 1.1 Experiment: Base & Finetune Training Metrics

**Dataset:** HAM10000  
**Epochs completed:** 15% of planned training  

### Metrics
- **Accuracy:** 66.5%  
- **AUC:** 0.923  
- **Loss:** 0.974  

### Observations
- Accuracy is moderate, but AUC is very high.  
- The model ranks a randomly chosen positive case higher than a negative case **92.3% of the time**.  
- The discrepancy between accuracy and AUC is likely due to **class imbalance** in HAM10000, which causes the model to struggle with underrepresented classes.  

### Next Steps
1. Run the same training using **EfficientNetB0** as the backbone to compare performance.  
2. Experiment with **different numbers of epochs** to see if accuracy improves.  
3. Explore **quantization** to prepare for potential deployment optimizations.