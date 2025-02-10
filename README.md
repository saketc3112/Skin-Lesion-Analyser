# **Skin Lesion Analyser**  

This repository contains the implementation of our paper:  
**[Skin Lesion Analyser: An Efficient Seven-Way Multi-Class Skin Cancer Classification Using MobileNet](https://link.springer.com/chapter/10.1007/978-981-15-3383-9_15)**  

## **Overview**  

Skin cancer is a growing global health concern, with **123,000 melanoma** and **3,000,000 non-melanoma cases** reported annually worldwide. Excessive exposure to ultraviolet rays is a major risk factor. **Early detection** is crucial, as the **five-year survival rate for melanoma is 99%** when diagnosed at an early stage. However, accurate diagnosis remains challenging for dermatologists, necessitating **automated, AI-driven solutions** to improve detection accuracy. This study presents an **efficient deep learning-based skin cancer classification model** that surpasses previous approaches and expert dermatologists in key evaluation metrics.  

### **Methodology**  

✅ **Model Architecture:** MobileNet pretrained on **1.28 million** images from the **2014 ImageNet Challenge**  
✅ **Dataset:** Fine-tuned on **10,015 dermoscopy images** from the **HAM10000 dataset**  
✅ **Evaluation Metrics:**  
- **Overall accuracy:** **83.1%** across **7 skin lesion classes**  
- **Top-2 accuracy:** **91.36%**  
- **Top-3 accuracy:** **95.34%**  
- **Weighted Precision:** **89%**  
- **Weighted Recall:** **83%**  
- **Weighted F1-score:** **83%**  

This method provides a **robust AI-powered tool** to assist dermatologists in **critical decision-making** for early-stage skin cancer detection.  

<p align="center">
  <img width="800" alt="Skin Lesion Analyser" src="https://github.com/user-attachments/assets/f981993b-4b9a-4b5b-9c7c-b28700b3ffca">
</p>  

---

## **Citation**  

If you find this work useful, please consider citing our paper:  

```bibtex
@inproceedings{chaturvedi2021skin,
  title={Skin lesion analyser: an efficient seven-way multi-class skin cancer classification using MobileNet},
  author={Chaturvedi, Saket S and Gupta, Kajol and Prasad, Prakash S},
  booktitle={Advanced machine learning technologies and applications: proceedings of AMLTA 2020},
  pages={165--176},
  year={2021},
  organization={Springer}
}
