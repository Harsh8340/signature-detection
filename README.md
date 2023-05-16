Deep Learning based Signature Detection and Verification
Signature verification systems are an essential part of most business practices. A significant amount of time and skillful resources could be saved by automating this process. This project demonstrates the implementation of an end-to-end signature verification system.

From the document the user selected, the signatures are extracted using YOLOv5. In real-world documents, there would be noise artifacts such as printed text, stamps etc which might seriously affect the performance of signature verification task. Thus a CycleGAN based noise cleaning method is added to tackle this. The cleaned signature is verified using a VGG16 based feature extractor, similar to Siamese Networks.

This project is based on these two papers [1] and [2].
[1] studies the usage of different object detection algorithms for signature detection and the results indicate that YOLOv5 outperforms all other models for the signature detection task. [2] provides a CycleGAN based approach to clean noise artifacts from signatures that are present in real-world documents and methods to perform signature validation using Representation learning.

This project has been trained and tested on signature datasets (Tobacco 800 and Kaggle Signature Dataset).

Model weights and data is not added, will update soon. :)

Workflow
The project works in three phases. Pipeline

Signature Detection
DetectionExample
Once the document to run inference is selected by the user, a YOLOv5 model will be run to detect and crop the signatures present in the document. YOLO model is trained using custom dataset created from Tobacco 800 dataset. The notebook to convert Tobacco 800 dataset to YOLOv5 format could be found here

Signature Cleaning
'Gan Example Real'
Signatures on real-world documents often contains noise artifacts like stamps/seals, text and printed lines. These noise artifacts might affect the signature verification process. A noise cleaning method based on CycleGAN will be performed on the detected signatures to generate noise free signatures. The CycleGAN model is trained using Kaggle Signature Dataset. Noisy signatures are generated from the dataset using OpenCV.This notebook contains code to generate noisy images and to convert the dataset to CycleGAN input format.

Signature Verification
vgg_model_working
In the final phase, a VGG16 based feature extractor is used. The model is fine-tuned on the Kaggle Signature dataset to learn the writer independent signature representations, thus new user signatures can be added to the system without re-training the model.
The cleaned image from the document and the reference signature (anchor image) of the user is fed into the model. The model outputs a vector (feature) that represents the signature. The features extracted from both the anchor image and cleaned signature from the document is used to compute the cosine similarity. Cosine similarity tells us how similar these two images are and it's value ranges from 0 to 1. From my experimentation, I have found out that for a matching signature pair the values are close to 1 and for a non-matching signature pair, the values are below 0.7. So I recommend a cosine similarity score of 0.8 as a threshold value to decide whether the signatures are a match or not. A more detailed take on the thresholds could be found on the recommendations section.
