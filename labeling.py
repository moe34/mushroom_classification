import matplotlib.pyplot as plt
import os
import cv2
import random
import numpy as np

def create_training_data(datadir, categories, img_size, training_data):
    for class_num, category in enumerate(categories):
        path = os.path.join(datadir, category)
        # os.remove(path+"/.DS_Store")
        for img_name in os.listdir(path): #os.listdirで指定したディレクトリの中身を一覧で取得
            # print(os.path.join(path, img_name))
            try:
                img_array = cv2.imread(os.path.join(path, img_name))
                # print(img_array,img_name)
                img_resize_array = cv2.resize(img_array, (img_size, img_size))
                training_data.append([img_resize_array, class_num])
            except Exception as e:
                print("skip " + img_name)

if __name__ == "__main__":
    x_train = [] #image data
    y_train = [] #label data
    training_data = [] #training_data
    create_training_data(datadir="dataset/", categories=["hiratake","tsukiyotake"], img_size=128, training_data=training_data)
    random.shuffle(training_data)

    for feature, label in training_data:
        x_train.append(feature)
        y_train.append(label)
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    for i in range(4):
        print("label of " + str(i), y_train[i])
        print("img of " + str(i), x_train[i])
        plt.subplot(2, 2, i+1)
        plt.axis("off")
        plt.title(label = "hiratake" if y_train[i]==0 else "tsukiyotake")
        img_array = cv2.cvtColor(x_train[i], cv2.COLOR_BGR2RGB)
        plt.imshow(img_array)

    plt.show()