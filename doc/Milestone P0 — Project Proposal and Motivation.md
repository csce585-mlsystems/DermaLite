# Efficient Model Quantization and Deployment on Apple Silicon with MLX  
**Theodore Villalva, DJ Ravenell, Ryan Caudill**  
Department of Computer Science  
University of South Carolina  

---

## Project Proposal

### Problem  
Deep learning models such as large language models (LLMs) often require substantial memory and computational resources, making them difficult to deploy on consumer-grade hardware or low-power devices. This limits their accessibility and practical usage outside of high-performance data centers. Efficient quantization and deployment pipelines can significantly reduce model size and inference latency while maintaining accuracy, enabling real-time applications on Apple Silicon and other edge devices.

---

### Literature Review  
To ground this project in existing research, we will examine prior work focusing on Apple Silicon performance and efficient model deployment:  

- *Profiling Apple Silicon Performance for ML Training* (Feng et al., 2025) explores the performance differences between Apple Silicon and NVIDIA GPUs, identifying memory architecture and kernel-launch overheads as key bottlenecks. [^1]  
- *Towards Large-scale Training on Apple Silicon* (van der Ouderaa et al., 2025) investigates the feasibility of training large language models on clusters of consumer-grade Apple hardware, proposing optimizer and hardware-utilization strategies. [^2]  

These works provide a strong technical foundation for our exploration of quantization and deployment within Apple’s MLX ecosystem.

---

### Data  
Since this project focuses on **quantization, benchmarking, and deployment**, we will use well-established **public datasets** rather than collect new data. These datasets are computationally efficient yet representative of real-world use cases:

- **CIFAR-10** — For ResNet quantization, providing a lightweight yet standard benchmark for image-classification accuracy versus inference speed.  
- **WikiText-2** — For LLaMA-style transformer quantization, enabling evaluation of model perplexity and efficiency in NLP tasks.  

If time permits, both datasets will be incorporated to demonstrate quantization across **computer vision and NLP** domains, offering a broader perspective on model optimization trade-offs.

---

### Method  
We will implement **post-training quantization** using Apple’s MLX framework, focusing on reducing precision (e.g., `int8` and `int4`) for both convolutional and transformer-based architectures.  

**Implementation Plan:**  
- Start with pretrained **ResNet** (for CIFAR-10) and **LLaMA-like models** (for WikiText-2).  
- Apply post-training quantization to convert full-precision weights to lower-bit representations.  
- Benchmark models in terms of **accuracy**, **memory footprint**, and **inference speed** on Apple Silicon hardware.  
- Package the quantized models into a lightweight inference service (e.g., via FastAPI) to demonstrate real-world deployability and API-based access.  

This workflow highlights both **optimization efficiency** and **production readiness**, aligning with the project’s practical deployment focus.

---

### Evaluation  
We will evaluate the impact of quantization through both quantitative metrics and visual performance analyses.

**Quantitative Evaluation:**  
- Classification accuracy (for CIFAR-10) — comparison of pre- and post-quantization performance.  
- Perplexity (for WikiText-2) — measuring degradation in language modeling tasks.  
- Inference latency — average time per forward pass on Apple Silicon.  
- Throughput — number of inferences processed per second.  
- Memory efficiency — reduction in model size and peak memory usage.  

**Qualitative Evaluation:**  
- Plot accuracy vs. model size and accuracy vs. inference time to visualize trade-offs.  
- Compare different quantization levels (int8, int4) to show how precision impacts deployment efficiency.  

These evaluations will demonstrate the real-world effectiveness of MLX quantization on Apple Silicon and the engineering trade-offs involved.

---

### Expected Outcomes  
By project end, we aim to:  
- Develop a **reproducible quantization and benchmarking pipeline** for MLX.  
- Quantify **accuracy-throughput trade-offs** on Apple Silicon.  
- Deliver a deployable **FastAPI-based inference service** for quantized models.  

This work contributes toward efficient on-device AI deployment and helps bridge the gap between high-performance research models and consumer-grade hardware.

---

### References  
[^1]: Feng, D., Xu, Z., Wang, R., Lin, F. X. (2025). *Profiling Apple Silicon Performance for ML Training.* arXiv preprint arXiv:2501.14925.  
[^2]: van der Ouderaa, T. F. A., Baioumy, M., Beton, M., Howes, S., Vrabie, G., Cheema, A. (2025). *Towards Large-scale Training on Apple Silicon.* Workshop ES-FoMo @ ICML 2025.  

