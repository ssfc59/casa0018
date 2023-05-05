# Image Classification of Apples, Oranges and Other Common Produce

Edge Impulse link: <https://studio.edgeimpulse.com/studio/201769>

Github link: <https://github.com/ssfc59/casa0018/tree/main/Assessment>
![image](https://user-images.githubusercontent.com/114293506/236506236-32d46b6c-6f33-498a-b49b-06c392efa85b.png)

## 1\. Introduction:
My project's idea was to create a deep learning model with Edge Impulse that could differentiate between apples, oranges and other produce with the camera on my mobile phone. I want to deploy in a market setting to help people's shopping or inventory sorting process in produce stores.

Fruit classification is a rising topic in computer and machine vision (Gill et al., 2022). It is useful in taking inventory and sales at grocery stores. On an industry level, it is helpful in automating fruit sorting systems in factories and for supply chain logistics. There are preexisting projects that identify fruits with Arduino and Tensorflow, that operate with an Arduino colourimeter and proximity sensor. My project's mobile deployment of a grocery produce image classification model makes it more accessible and usable at a smaller scale: for example, it could help cashiers to correctly label items, or help visually impaired people correctly identify the items they want by sweeping their phone camera over the produce section. 

After gathering my dataset, using Edge Impulse to create and finetune my model architecture, and deploying it on my mobile phone and testing it in different locations, I have created a image classification model that can differentiate between images of apples, oranges and other common produce in a store setting with 81% accuracy.

## 2\. Research Question:
Can I create a sensor that can accurately differentiate between images of apples, oranges and other common produce?

## 3\. Application Overview:
My application components were my uploaded data, Edge Impulse, and my own mobile phone, an iPhone 12.
![image](https://user-images.githubusercontent.com/114293506/236500017-4c566a63-0086-4d0e-9670-02934bfc6c3b.png)

Figure 1: Components of my deep learning model
The steps I took were:
1. Choose dataset
 2. Input data into Edge Impulse to create a quantised model that can be deployed to other devices
3. Experiment to create the most effective model architecture
4. Real world deployment

4. Data:
The training and testing data I collected was split into 3 labels:
- 1,252 images each of apples
- 1,262 images of oranges
- 1,231 images of other produce commonly found in a supermarket, which was fed with 100 images each of 12 different fruits and vegetables: 
1. Bananas
2. Bell Peppers
3. Grapes
4. Kiwis
5. Lemons
6. Mangoes
7. Onions
8. Pears
9. Potatoes
10. Pineapples
11. Tomatoes
12. Watermelons

The data was collected from multiple Kaggle datasets. The majority of my data was from the apples2oranges Kaggle dataset, which was one of UC Berkeley's Office CycleGAN (Generative Adversarial Network, another type of machine learning framework) datasets. It contained 2,528 256 x 256 pixel images of both individual and multiple apples and oranges in a general setting. 
![image](https://user-images.githubusercontent.com/114293506/236500644-49be8f31-0502-4131-a43e-59c6b0c2b50e.png)
![image](https://user-images.githubusercontent.com/114293506/236500693-825a73a8-8e57-4f6b-88c2-1f302b56b3b4.png)

Figure 2. Example pictures of data from the apples2oranges dataset
The data I imported into the Other Produce label was gathered from 4 different Kaggle datasets, which also contained images of produce in a realistic setting:
- Fruits-262
- Vegetable Image Dataset
- Fruit classification (10 Class)
- Fruits and Vegetables Image Recognition Dataset

![image](https://user-images.githubusercontent.com/114293506/236501151-f79cbc2f-b0f4-4e77-8369-f5dc9cf78cf3.png)

Figure 3. Example pictures of data informing my “Other Produce” data
In this data gathering process, I looked through other datasets, such as the Fruits360 dataset, that contained images of individual fruits and vegetables against a blank background. 

![image](https://user-images.githubusercontent.com/114293506/236501287-e81f3d12-9a07-4b43-95df-07ee675ed1d2.png)
![image](https://user-images.githubusercontent.com/114293506/236501311-13d8b60e-fd96-4453-9762-c0d15078e51c.png)

Figure 4. Data from the Fruits360 dataset
However, as I wanted to deploy my model in a realistic grocery store setting that contains scenes with busy backgrounds and multiple fruits placed together, I wanted to enable my model to classify both singular and multiples of different produce items against a realistic backdrop. Therefore, I found this dataset to be unsuitable for my purposes, as a model trained with singular fruit images in a uniform background and setting would limit its classification accuracy in my intended deployment setting. 
In total, I used 3,475 items in my image classification model. Afterwards, I uploaded all my datasets into a new project on Edge Impulse to begin creating my model.

## 5. Model architecture: Trials and Results

Edge Impulse allows users to build machine learning models with different data processing tools and pre-trained neural network architectures. 
![image](https://user-images.githubusercontent.com/114293506/236501512-d5b7fbec-3f45-406a-9c69-9d2843da7514.png)
Figure 5. The different blocks of Impulse design within Edge Impulse
To determine the best model architecture for my project, I ran 7 trials and adjusted their configurations based on the previous trials results. I documented each trial in the spreadsheet below along with my observations and changes.

![image](https://user-images.githubusercontent.com/114293506/236501833-a53c1083-8168-4349-ae08-24ed2ed3dd0f.png)

Figure 6. Spreadsheet documenting my model experimentation process. Variables I changed from the previous experiment are highlighted in yellow, and the best performing trials are highlighted in green. 

I adjusted all suitable customisation options during my experiments to create the best performing model. 

The factors I paid most attention to while refining my model were:
- the number of images labeled uncertain
- image size
- color depth
- taking precautions against extreme overfitting

### 5.1 Best Performing models:
I found that the best performing model to be Trial 4, which utilised the MobileNetV2 160x160 0.35 model, closely followed by Trial 6, which utilised the MobileNetV2 96x96 0.35 model. 
![image](https://user-images.githubusercontent.com/114293506/236503423-184bd86d-a90c-419d-a9bc-ffed609f3d4e.png)

Figure 7. Table comparing the training and testing results of the 2 best performing trials

After seeing the 89.60% training data accuracy versus the 50.93% testing accuracy on Trial 3, I decreased the image size and model, number of epochs, and dropout rate to prevent such overfitting results in my next trials. Consequently, the results of Trial 4 showed a high and similiar testing and training accuracy of 75.60% and 77.78% respectively, which showed that the model did not overfit. Interestingly, this trial was the only neural network architecture to more accurately classify the testing data compared to the training data. 

On the other hand, Trial 6 had a good and accurate performance with both testing and training accuracy above 80%. Since Trial 6 had less uncertain labels compared to Trial 4 in the testing accuracy, I chose to deploy Trial 6 to my mobile phone.

From these results, I found these factors to be the most crucial to the model’s accuracy:
- RGB colour depth instead of greyscale
- Transfer learning model instead of convolutional neural network model
- Data augmentation
- Overfitting preventions: increased dropout rate and decreased number of neurons in the final layer from the default standards
As there were well-performing models with both 160x160 and 96x96 pixel image sizes, I did not think that image size was a major contributing factor to accuracy for my purposes.

## 6. Results and Observations:
I deployed the model created in Trial 6 to my iPhone in several different real life settings: my kitchen, the UCL East Shop, and Waitrose. It performed well overall in this variety of settings, lightings and backgrounds.
![image](https://user-images.githubusercontent.com/114293506/236503113-398ec9b1-0f52-4c3d-9f9e-2d59fae115e6.png)

Figure 8. Screenshots of deployment in my kitchen, the UCL East Shop and Waitrose respectively.

The losses in accuracy can mostly be attributed to the “Other Produce” category, as its data are misclassified at much higher rates compared to those from other categories. The graph of Trial 6’s testing results show that test data in the “Other Produce” category was accurately classified 70.6% of the time, compared to the 88.6% and 85.3% classification accuracy of “Apple” and “Orange” respectively. Moreover, 18.1% of “Other Produce” data was misclassified as “Orange”. Other red-coloured produce, such as red onions, red peppers and cut watermelons were more likely to be misclassified as apples. 
![image](https://user-images.githubusercontent.com/114293506/236503589-bb5fd60c-796e-420c-8d8c-2d96505b0884.png)

Figure 9. Examples of misclassified “Other Produce” data as “Apple”

Meanwhile, other yellow or orange coloured produce such as mangoes, lemons, and tomatoes are more likely to be misclassified as oranges.
![image](https://user-images.githubusercontent.com/114293506/236503759-8de12d04-2507-4315-a56e-f0d222f6b7dc.png)

Figure 10. Examples of misclassified “Other Produce” data as “Orange”



### 6.1 Factors affecting deployment accuracy
Based on the model’s performance on both the test data and in real life, I identified some performance trends and documented them in a table.
![image](https://user-images.githubusercontent.com/114293506/236503931-dc5f0f64-41bc-497a-8bec-457a5ec7f3e4.png)
Figure 11. Table documenting model performance depending on the camera's perspective and proportion of produce items
- Multiples of the same produce item: The model is likely to perform more accurately when shown an image containing multiples of the same produce item, as the more of them there are, the more likely one produce item is to be correctly classified. 
- Multiple different produce items: Conversely, it performs fairly when shown images containing multiple different produce items, often giving incorrect or “uncertain” labels. 
- Close up images: The model also performs better with close up images of the produce, as it is more able to identify its distinctive features. 
- Angled views: The model’s performance did not vary much between images taken from a top or side view.

### 6.2 Specific individual factors
I made another table to analyse the category specific factors affecting classification accuracy.
![image](https://user-images.githubusercontent.com/114293506/236504424-083da1fb-4b2e-40ec-94ee-859d81d560c4.png)

Figure 12. Table examining the importance of factors that facilitate accurate identification of produce

#### Expected colour: 
Expected colour was the biggest determining factor for accurate produce identification. Trained on my chosen dataset, my model strongly associates red-coloured items with apples, orange-coloured items with oranges, and other coloured items with other produce items. This relationship has the most weighing on image classification of apples: while this green apple is incorrectly classified, it is able to correctly classify this red apple through its packaging. 

![image](https://user-images.githubusercontent.com/114293506/236504596-e70aeb31-0080-4db9-a69f-2c0fd994018d.png)

Figure 13. Misclassification of green apple vs correct classification of red apple through its packaging in Waitrose

![image](https://user-images.githubusercontent.com/114293506/236504717-3f8c522a-893d-41d8-9032-30735f999794.png)

Figure 14. Orange coloured tomato incorrectly classified as “Orange”

#### Produce orientation: 
The model’s accuracy also varies with the orientation of the produce item in the analysed image. In this example, after I turn a correctly classified apple 90 degrees, the model classified the apple as an orange instead. 
![image](https://user-images.githubusercontent.com/114293506/236505060-61474c34-1a01-45bd-b54d-5b8dc840b5b2.png)

Figure 15. Screenshot of differing results upon turning an apple 90 degrees

Perhaps my model associates more vertically elongated shapes with apples, and more squat and circular shapes with oranges.

#### Best at classifying oranges:
Overall, the expected colour, usual orientation and presence of packaging did not affect the classification accuracy of oranges specifically. Therefore, I conclude that my model can classify oranges more reliably than all other produce items.

## 7. Conclusions:
To conclude, I have carefully adjusted this transfer learning model so it can consistently differentiate between apples, oranges and other common produce in a real-world setting. Throughout this project, I have reflected on its limitations and possible ways to improve its performance:

 Firstly, perhaps I could improve the dataset for the “Other Produce” category by using images of a standard size, like the uniform 256 x 256 pixel images in apples2oranges, instead of gathering images of unset sizes from different datasets.

Also, since the model’s accuracy greatly depends on the colour of the produce item, equating red produce items with “Apple” and orange produce items with “Orange”, it limits its accuracy in classifying differently coloured apples like green apples. To mitigate this, I could add more images of fruits of different varieties and colours to the training data of this model to improve its classification accuracy. I could also find more areas for improvement by asking my target groups, people who work in supermarkets, or people who are visually impaired, to test and provide feedback on it.

In the future, I can develop this further by adding training and testing data, adding more classification categories to identify more produce apart from apples and oranges, or adding a “mixed produce” section to classify images that contain multiple different kinds of produce in them. I could also further modify the model architecture on TensorFlow to get around Edge Impulse’s 20 minute load time limit.


## 8. References:
Singh Gill, H., Ibrahim Khalaf, O., Alotaibi, Y., Alghamdi, S., et al. (2022) Fruit Image Classification Using Deep Learning. Computers, Materials & Continua. [Online] 71 (3), 5135–5150. Available from: https://doi.org/10.32604/cmc.2022.022809.

### Kaggle datasets:
- Apple2orange: https://www.kaggle.com/datasets/balraj98/apple2orange-dataset
- (Fruit classification (10 Class)) https://www.kaggle.com/datasets/karimabdulnabi/fruit-classification10-class
- Fruits and Vegetables Image Recognition Dataset: https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition
- Fruits-262: https://www.kaggle.com/datasets/aelchimminut/fruits262
- Fruits 360: https://www.kaggle.com/datasets/moltean/fruits/code
- Vegetable Image Dataset: https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset

## Declaration of Authorship:
I, Sophia Chong, confirm that the work presented in this assessment is my own. Where information has been derived from other sources, I confirm that this has been indicated in the work.

Sophia Chong 5/5/2023






