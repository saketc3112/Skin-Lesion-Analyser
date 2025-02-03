# Skin Lesion Analyser 

This repository contains code for the paper [Skin Lesion Analyser: An Efficient Seven-Way Multi-Class Skin Cancer Classification Using MobileNet](https://link.springer.com/chapter/10.1007/978-981-15-3383-9_15)

### Overview
Skin cancer is an emerging global health problem with 123,000 melanoma and 3,000,000 non-melanoma cases worldwide each year. The recent studies have reported excessive exposure to ultraviolet rays as a major factor in developing skin cancer. The most effective solution to control the death rate for skin cancer is a timely diagnosis of skin lesions as the five-year survival rate for melanoma patients is 99% when diagnosed and screened at the early stage. Considering an inability of dermatologists for accurate diagnosis of skin cancer, there is a need to develop an automated efficient system for the diagnosis of skin cancer. This study explores an efficient automated method for skin cancer classification with better evaluation metrics as compared to previous studies or expert dermatologists. We utilized a MobileNet model pretrained on approximately 1,280,000 images from 2014 ImageNet Challenge and finetuned on 10,015 dermoscopy images of HAM10000 dataset employing transfer learning. The model used in this study achieved an overall accuracy of 83.1% for seven classes in the dataset, whereas top2 and top3 accuracies of 91.36% and 95.34%, respectively. Also, the weighted average of precision, weighted average of recall, and weighted average of f1-score were found to be 89%, 83%, and 83%, respectively. This method has the potential to assist dermatology specialists in decision making at critical stages. We have deployed our deep learning model at https://saketchaturvedi.github.io as Web application.

<img width="909" alt="Screenshot 2025-02-03 at 5 42 49 PM" src="https://github.com/user-attachments/assets/f981993b-4b9a-4b5b-9c7c-b28700b3ffca" />

## Web Application

https://saketchaturvedi.github.io

## Citation

If you find our code useful, please cite our paper. :)

`@inproceedings{chaturvedi2021skin,
  title={Skin lesion analyser: an efficient seven-way multi-class skin cancer classification using MobileNet},
  author={Chaturvedi, Saket S and Gupta, Kajol and Prasad, Prakash S},
  booktitle={Advanced machine learning technologies and applications: proceedings of AMLTA 2020},
  pages={165--176},
  year={2021},
  organization={Springer}
}`
