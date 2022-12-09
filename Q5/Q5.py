import tensorflow as tf
import tensorflow_addons as tfa
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import glob
import random
from PIL import Image
import numpy as np


class Question5:
    def __init__(self) -> None:
        pass

    def showInference(self, imgPath):

        if imgPath == None:
            print("please load the image which want to predict")
            return


        print("predict start...")

        import os
        os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
        model = tf.keras.models.load_model("./model/model_focol_loss.h5")

        img = Image.open(imgPath)
        img = img.resize((224, 224))

        img = np.array(img).astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        pred = model.predict(img)

        if pred[0][0] > 0.5:
            predOutcome = "cat"
        else:
            predOutcome = "dog"
        
        print(predOutcome)
        print("predict done...")

        return predOutcome

    def showImages(self):
        fig, axes = plt.subplots(1, 2, figsize=(16, 8), num="Resized images in inference dataset")

        catInferenceDataset = glob.glob(".\\inference_dataset\\Cat\\*.jpg")
        dogInferenceDataset = glob.glob(".\\inference_dataset\\Dog\\*.jpg")

        imgCat = Image.open(random.choice(catInferenceDataset))
        imgCat = imgCat.resize((224, 224))

        imgDog = Image.open(random.choice(dogInferenceDataset))
        imgDog = imgDog.resize((224, 224))

        axes[0].imshow(imgCat)
        axes[0].set_title("Cat")
        axes[0].axis("off")

        axes[1].imshow(imgDog)
        axes[1].set_title("Dog")
        axes[1].axis("off")

        plt.show()

    def showDistribution(self):

        catTrainingDataset = glob.glob(".\\training_dataset\\Cat\\*.jpg")
        dogTrainingDataset = glob.glob(".\\training_dataset\\Dog\\*.jpg")

        left = np.array(["cat", "dog"])
        height = np.array([len(catTrainingDataset), len(dogTrainingDataset)])

        plt.figure(num="Class Distribution")
        # show the number on the bar
        for a, b, i in zip(left, height, range(len(left))):
            plt.text(a, b, height[i], ha='center', fontsize=10)

        plt.bar(left, height)

        plt.title("Class Distribution")
        plt.ylabel("Number of images")
        plt.show()

    def showModelStructure(self):
        import os
        os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
        model = tf.keras.models.load_model("./model/model_focol_loss.h5")
        model.summary()

    def showComparison(self):
        # make a comparison picture
        # self.compareModels()

        image = plt.imread("model_comparison.png")

        # Display the image
        plt.figure(num="Model Comparison")
        plt.axis("off")
        plt.imshow(image)
        plt.show()


    def compareModels(self):
        import os
        os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
        model_binary_cross_entropy = tf.keras.models.load_model("./model/model_binary_cross_entropy.h5")
        model_focol_loss = tf.keras.models.load_model("./model/model_focol_loss.h5")
        
        datagen = ImageDataGenerator(rescale=1./255.)

        validation_generator=datagen.flow_from_directory(
        directory = "./validation_dataset/",
        class_mode = "categorical",
        batch_size = 32,
        target_size = (224, 224))

        loss_binary_cross_entropy, accuracy_binary_cross_entropy = model_binary_cross_entropy.evaluate(validation_generator)
        loss_focol_loss, accuracy_focol_loss = model_focol_loss.evaluate(validation_generator)
        


        left = np.array(["Binary Cross Entropy", "Focal Loss"])
        height = np.array([round(accuracy_binary_cross_entropy * 100, 1), round(accuracy_focol_loss * 100, 1)])

        # show the number on the bar
        for a, b, i in zip(left, height, range(len(left))):
            plt.text(a, b, height[i], ha='center', fontsize=10)

        plt.bar(left, height)

        plt.title("Accuracy Comparison")
        plt.ylabel("Accuracy(%)")

        plt.savefig("model_comparison.png")
        plt.show()




if __name__ == "__main__":
    # for testing
    Q5 = Question5()
    Q5.showImages()
    Q5.showDistribution()
    Q5.showModelStructure()
    Q5.compareModels()
    Q5.showComparison()
    