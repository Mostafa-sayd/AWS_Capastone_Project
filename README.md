
# Amazon Bin Image Dataset(ABID) Challenge

## Project Overview



The  Huge companies  like Amazon all over the world usually move objects from place to another and those companies got many objects  to be moved so they move the objects using bins which have many objects inside and each object has a number so the company could keep track of the object and make sure that everything is going well .

The Amazon Bin Image Dataset contains 50,000 images and metadata from bins of a pod in an operating Amazon Fulfillment Center. The bin images in this dataset are captured as robot units carry pods as part of normal Amazon Fulfillment Center operations. This dataset can be used for research in a variety of areas like computer vision, counting genetic items and learning from weakly-tagged data.


## Problem Statement


The Amazon Bin Image Dataset contains images and metadata from bins of a pod in an operating Amazon Fulfillment Center. The bin images in this dataset are captured as robot units carry pods as part of normal Amazon Fulfillment Center operations  
when we want the bins double-checked. The task is sometimes very challenging because of heavy occlusions and a large number of object categories.

 We would like to open a new challenge in order to attract talented researchers in both academia and industry for these tasks. As a starting point, we provide baseline methods and pre-trained models for two tasks, counting and object verification tasks.
This is a simple task where you are supposed to count every object instance in the bin. We count individual instances separately, which means if there are two same objects in the bin, you count them as two.


As we could see, the problem is an Image Classification problem  as the image is provided to build the ML/DL model  to identify the number of objects in each bin. 


## Project filesProject 

consist of multiple files:

sagemaker.ipynb -- main project file. Entrypoint

train_model.py -- python script for tuning the network. Can be used from Sagemaker or as standalone application

inference.py -- python script for running model inference

file_list.json -- queried for the database to download only part of the dataset

## Evaluation Metrics


I would use the evaluation at the end of each epoch and see the progress of the model  using both Accuracy and RMSE for Evaluation Metrics.

I used it  at the end of each epoch to show the improvement of the model .


## Datasets and Inputs



These are some typical images in the dataset. A bin contains multiple object categories and various number of instances. 

The corresponding metadata exists for each bin image and it includes the object category identification(Amazon Standard Identification Number, ASIN) which contains more than 500,000 image and metadata, quantity, size of objects, weights, and so on. 
The size of bins are various depending on the size of objects in it. The tapes in front of the bins are for preventing the items from falling out of the bins and sometimes it might make the objects unclear.

Objects are sometimes heavily occluded by other objects or limited viewpoints of the images.

[kaggle](https://www.kaggle.com/datasets/dhruvildave/amazon-bin-image-dataset)

The dataset contains 5 classes which identify each object in the picture 


MetaData:
'''
{
    "BIN_FCSKU_DATA": {
        "B00CFQWRPS": {
            "asin": "B00CFQWRPS",
            "height": {
                "unit": "IN",
                "value": 2.399999997552
            },
            "length": {
                "unit": "IN",
                "value": 8.199999991636
            },
            "name": "Fleet Saline Enema, 7.8 Ounce (Pack of 3)",
            "normalizedName": "(Pack of 3) Fleet Saline Enema, 7.8 Ounce",
            "quantity": 1,
            "weight": {
                "unit": "pounds",
                "value": 1.8999999999999997
            },
            "width": {
                "unit": "IN",
                "value": 7.199999992656
            }
        },
        "ZZXI0WUSIB": {
            "asin": "B00T0BUKW8",
            "height": {
                "unit": "IN",
                "value": 3.99999999592
            },
            "length": {
                "unit": "IN",
                "value": 7.899999991942001
            },
            "name": "Kirkland Signature Premium Chunk Chicken Breast Packed in Water, 12.5 Ounce, 6 Count",
            "normalizedName": "Kirkland Signature Premium Chunk Chicken Breast Packed in Water, 12.5 Ounce, 6 Count",
            "quantity": 1,
            "weight": {
                "unit": "pounds",
                "value": 5.7
            },
            "width": {
                "unit": "IN",
                "value": 6.49999999337
            }
        },
        "ZZXVVS669V": {
            "asin": "B00C3WXJHY",
            "height": {
                "unit": "IN",
                "value": 4.330708657
            },
            "length": {
                "unit": "IN",
                "value": 11.1417322721
            },
            "name": "Play-Doh Sweet Shoppe Ice Cream Sundae Cart Playset",
            "normalizedName": "Play-Doh Sweet Shoppe Ice Cream Sundae Cart Playset",
            "quantity": 1,
            "weight": {
                "unit": "pounds",
                "value": 1.4109440759087915
            },
            "width": {
                "unit": "IN",
                "value": 9.448818888
            }
        }
    },
    "EXPECTED_QUANTITY": 3
}
'''

##  Algorithm

Solution would be to build a Deep Learning model which would help to count the objects in each picture by using pre pre-trained model  like Resnet as we used before in the previous project. In my case I used resnet18 .

ResNet model is widely used for image classification which is pretrained and can be customized in order to categorize images from different use cases. To adapt this pretrained model to our use case, different training jobs will be launched in AWS SageMaker. In addition, hyperparameters tuning jobs has been launched in order to find the most appropriate combination of hyperparameters for our use case.

As mentioned in the hyperparameter tuning section below we would fine-tune some parameters like learning rate and batch size as well as the number of epochs . 


## Model Training and Refinement

Building a Deep Learning model which would help to count the objects in each picture by using pre pre-trained model  like Resnet as we used before in the previous project. In my case I used resnet18 .

As stated before, I planned to use a ResNet neural network to train the model
As a base I used this Python training script, which is the one I implemented for the "Image Classification" project of this course (file train_model .py ). I adapted the number of classes (from 133 to 5) and configured the transformation part in order to deal with the new set of images (i.e. resizing). 

Then I launched this script through the Jupyter Notebook. However, the results, as can be seen on this screenshot,  not very promising, with a RMSE of 1.48 and an accuracy of 30.20%.



I decided it was high time to tune the hyperparameters in order to find a better combination of them which allow me to obtain a more precise model. Specifically, these hyperparameters were tuned:

- The number of epochs (epoch) between (4,10 ) 
- The batch size (the number of images being trained on each iteration)( 32, 64, 256 )
- The learning rate ( lr )  (0.001,0.1)

After completing, the best hyperparameters combination was the following one: 

- Epochs: 5
- Batch Size: 32 
- Learning Rate: 0.1

With this combination, testing accuracy spiked to 32% and RMSE was 2.42




## Model deployment

After training model can be deployed and used from different AWS services. Deployment procedure is presented in notebook sagemaker.ipynb creating an endpoint to predict throw it then delete it to avoid cost .

After training model can be deployed and used from different AWS services. Deployment procedure is presented in notebook sagemaker.ipynb creating an endpoint to predict throw it then delete it to avoid cost   using interfec.py file for entery point.


