import numpy as np
import pandas as pd
from tensorflow import keras as kr
import matplotlib.pyplot as pl
import os
import cv2
import copy


def train_test_split(images, subParas, labels, trainRatio=0.99):
    amount = labels.shape[0]
    print(amount)
    array = np.arange(amount)
    trainIndices = np.random.permutation(array)[:int(amount*trainRatio)].tolist()
    testIndices = np.random.permutation(array)[-int(amount*trainRatio):].tolist()
    return images[trainIndices, :], subParas[trainIndices, :], labels[trainIndices, :], images[testIndices, :], subParas[testIndices, :], labels[testIndices, :]


def getCenterDeviationWithImage(image_origin, shouldTrim=True):
    bottomAcceptableMargin = 20
    y_start = 40 if shouldTrim else 0
    height = 80 if shouldTrim else 120
    y_end = y_start+height
    image_trimmed = image_origin[y_start: y_end, 0: 320]
    image_blurred = cv2.bilateralFilter(image_trimmed, 5, 200, 200)
    image_gray = cv2.cvtColor(image_blurred, cv2.COLOR_RGB2GRAY)
    # image_gray = cv2.cvtColor(image_trimmed,cv2.COLOR_RGB2GRAY)
    # dst = cv2.fastNlMeansDenoising(image_trimmed,h=20)
    # image_edge = cv2.Canny(dst, 150, 200)
    image_edge = cv2.Canny(image_gray, 150, 250)
    return image_edge

    left = 0
    right = 320
    gap = None
    # lines = cv2.HoughLinesP(image_edge, rho=1, theta=np.pi/360, threshold=50, minLineLength=50, maxLineGap=5)
    # if lines is not None:
    #     for line in lines:
    #         x1,y1,x2,y2 = line[0]
    #         if y1 > height - 20:
    #             if x1 < 160:
    #                 left = max(x1, left)
    #             else:
    #                 right = min(x1, right)
    #         if y2 > height - 20:
    #             if x2 < 160:
    #                 left = max(x2, left)
    #             else:
    #                 right = min(x2, right)
    #         # cv2.line(image_trimmed, (x1,y1), (x2,y2), (255, 0, 0), 2)
    #     if right == 320 and left == 0:
    #         return None, image_origin
    #     roadCenter = int((right + left)/2)
    #     roadHalfWidth = (right - left)/2
    #     gap = (160 - roadCenter) / roadHalfWidth
    #     cv2.circle(image_origin,(160,30+y_start),5,(255,0,0),thickness=1)
    #     cv2.circle(image_origin,(roadCenter,30+y_start),5,(0,0,255),thickness=1)
    #     return float(gap), image_origin
    # else:
    #     return None, image_origin
    countours, hierarchy = cv2.findContours(image_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # countours = np.array(countours)
    for countour in countours:
        countour = countour[:, 0, :]
        xs_bottom = countour[countour[:, 1] > height - bottomAcceptableMargin, 0]
        if len(xs_bottom) == 0:
            continue
        possibleLefts = xs_bottom[xs_bottom < 160]
        if len(possibleLefts) > 0:
            possibleLeft = possibleLefts.max()
            left = max(possibleLeft, left)
        possibleRights = xs_bottom[xs_bottom > 160]
        if len(possibleRights) > 0:
            possibleRight = possibleRights.min()
            right = min(possibleRight, right)

    if right == 320:
        right = 240
    # else:
    #     right_normalized = (right - 160) / 100
    # self.rightQueue = np.insert(self.rightQueue, 0, right)[:-1]
    if left == 0:
        left = 80
    # else:
    #     left_normalized = (160 - left) / 100.0
    # self.leftQueue = np.insert(self.leftQueue, 0, left)[:-1]
    # roadCenter = (right + left)/2
    # roadHalfWidth = (right - left)/2
    # gap = (160 - roadCenter) / roadHalfWidth
    # # Image center color_blue
    # cv2.circle(image_edge, (int(self.leftQueue.mean()), height-bottomAcceptableMargin),5,(255,0,0),thickness=3)
    # cv2.circle(image_edge, (int(self.rightQueue.mean()), height-bottomAcceptableMargin),5,(255,0,0),thickness=3)
    # cv2.circle(image_edge, (int((self.rightQueue.mean()+self.leftQueue.mean())/2), height-bottomAcceptableMargin),5,(255,0,0),thickness=3)
    # The center of the road  color_red
    # cv2.circle(image_edge, (int(roadCenter),height-5),5,(255,0,0),thickness=3)
    # cv2.circle(image_edge, (160,int(height-bottomAcceptableMargin)),5,(255,0,0),thickness=2)
    try:
        cv2.drawContours(image_edge, countours, 1, (255,0,0), 1)
    except:
        return image_edge
    return image_edge



class DeepModel():
    def __init__(self):
        # self.traceDataDirPath = '../data/trace1'
        self.traceDataDirPath = '../data/trace1'
        # load model
        self.modelPath = './continuousModel_rawImage.h5'
        self.model = self.__loadModel()


    # MARK: - Public Method

    def run(self):
        # images, subParas, labels = self.__loadAllTrainingDataSet()
        for images, subParas, labels in self.__loadOneTrainingDataSet():
            images_train, subParas_train, labels_train, images_test, subParas_test, labels_test = train_test_split(images, subParas, labels, trainRatio=0.99)
            # print(images_train.shape)
            # print(subParas_train.shape)
            # print(labels_train.shape)
            self.model.summary()
            self.model.fit(
                [images_train, subParas_train],
                # images_train,
                labels_train,
                batch_size=16,
                epochs=50,
                validation_split=0.1
            )
            print("Evaluate on test data")
            results = self.model.evaluate([images_test, subParas_test], labels_test, batch_size=16)
            print("test loss, test acc:", results)
            self.model.save(self.modelPath)


    def showVedio(self):
        fig = pl.figure()
        plottingImages = []
        images, subParas, labels = self.__loadOneTrainingDataSet()
        for image in images:
            image = getCenterDeviationWithImage(image, shouldTrim=False)
            # _, _ = self.getCenterDeviation(image)
            image_plot = pl.imshow(image, cmap='gray')
            plottingImages.append([image_plot])
        ani = animation.ArtistAnimation(fig, plottingImages, interval=50)
        pl.show()



    # MARK: - Private Method

    def __loadModel(self):
        if os.path.exists(self.modelPath):
            return kr.models.load_model(self.modelPath)
        # if os.path.exists('continuousModel_succeed.h5'):
        #     return kr.models.load_model('continuousModel_succeed.h5')
        else:
            image_inputs = kr.layers.Input(shape=(160, 320, 3), dtype=np.float, name='image')
            # image_inputs = kr.layers.Input(shape=(80, 320, 1), dtype=np.float, name='image')
            # image_inputs = layers.Input(shape=env.observation_spec['image']['shape'], dtype=np.float, name='image')
            image_layer = kr.layers.Conv2D(filters=64, kernel_size=4, strides=(2, 2), activation='relu', name='image_conv1')(image_inputs)
            image_layer = kr.layers.MaxPooling2D(pool_size=(4,4))(image_layer)
            image_layer = kr.layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu', name='image_conv2')(image_layer)
            image_layer = kr.layers.MaxPooling2D(pool_size=(4,4))(image_layer)
            image_layer = kr.layers.Conv2D(filters=128, kernel_size=3, strides=(1, 1), activation='relu', name='image_conv3')(image_layer)
            image_layer = kr.layers.MaxPooling2D(pool_size=(2,2))(image_layer)
            image_layer = kr.layers.Flatten(name='flattened')(image_layer)
            # image_layer = kr.layers.Dense(256, activation='relu', name='image_dense1')(image_layer)
            image_layer = kr.layers.Dense(64, activation='relu', name='image_dense2')(image_layer)
            image_layer = kr.layers.Dense(1, activation='tanh', name='image_dense3')(image_layer)

            # subPara_inputs = layers.Input(shape=env.observation_spec['subPara']['shape'], dtype=np.float, name='subPara')
            subPara_inputs = kr.layers.Input(shape=(1,), dtype=np.float, name='subPara')
            # subPara_dense = kr.layers.Dense(8, activation='tanh', name='subPara_dense')(subPara_inputs)

            common = kr.layers.concatenate([image_layer, subPara_inputs])

            num_actions = 1
            # common = kr.layers.Dense(128, activation="tanh", name='common_dense1')(common)
            # num_actions = env.action_spec['shape'][0]
            # action = kr.layers.Dense(num_actions, name='action_dense1')(common)
            common = kr.layers.Dense(num_actions, name='common_output')(common)

            model = kr.Model(inputs=[image_inputs, subPara_inputs], outputs=common)
            # model = kr.Model(inputs=image_inputs, outputs=common)
            model.compile(
                optimizer=kr.optimizers.Adam(learning_rate=1e-3),
                loss='mse',
                metrics=['mse']
            )
            # model.compile(
            #     optimizer=kr.optimizers.Adam(learning_rate=5e-3),
            #     loss=kr.losses.CategoricalCrossentropy(),
            #     metrics=[kr.metrics.CategoricalAccuracy()],
            # )
            return model


    def __loadOneTrainingDataSet(self):
        dataSetDirNames = list(filter(lambda name: '2021' in name, os.listdir(f"{self.traceDataDirPath}/")))
        dataSetDirNames = np.random.permutation(dataSetDirNames)
        for dataSetDirName in dataSetDirNames:
            images = []
            subParas = []
            labels = []
            steering_angle_last = 0.0
            throttle_last = 0.0
            speed_last = 0.0
            logPath = f"{self.traceDataDirPath}/{dataSetDirName}/driving_log.csv"
            print(f"start training {logPath} ...")
            logData = pd.read_csv(logPath, names=[
                'Center Image',
                'Left Image',
                'Right Image',
                'Steering Angle',
                'Throttle',
                'Break',
                'Speed'
            ])

            noneStraightDataCount = 0
            for index in logData.index[:-1]:
                logData.loc[index, 'Next Steering Angle'] = logData.loc[index+1, 'Steering Angle']
                if abs(logData.loc[index, 'Next Steering Angle']) <= 1e-2:
                    continue
                labels.append(logData.loc[index, 'Next Steering Angle'])
                # load images
                imageName = logData.loc[index, 'Center Image'].split('\\')[-1]
                imagePath = f"{self.traceDataDirPath}/{dataSetDirName}/IMG/{imageName}"
                image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # image = getCenterDeviationWithImage(image)
                images.append(image.astype("float32") / 255)

                # subPara = logData.loc[index, ['Steering Angle', 'Throttle', 'Speed']].values.ravel()
                # subPara[2] /= 30.5
                subPara = logData.loc[index, 'Steering Angle']
                subParas.append(subPara)
                noneStraightDataCount += 1

            # print(logData.groupby('Next Steering Angle').count())
            logData = logData.dropna()
            logData.to_csv(f"{self.traceDataDirPath}/{dataSetDirName}/new_log.csv")

            # add straight observations
            neededStraightAmount = noneStraightDataCount
            print(f"needed straight amount = {neededStraightAmount}")
            shuffledIndices = np.random.permutation(logData.index)
            straightCount = 0
            for index in shuffledIndices:
                if abs(logData.loc[index, 'Next Steering Angle']) > 1e-2:
                    continue
                labels.append(logData.loc[index, 'Next Steering Angle'])
                # load images
                imageName = logData.loc[index, 'Center Image'].split('\\')[-1]
                imagePath = f"{self.traceDataDirPath}/{dataSetDirName}/IMG/{imageName}"
                image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image.astype("float32") / 255)

                # subPara = logData.loc[index, ['Steering Angle', 'Throttle', 'Speed']].values.ravel()
                # subPara[2] /= 30.5
                subPara = logData.loc[index, 'Steering Angle']
                subParas.append(subPara)
                if straightCount < neededStraightAmount:
                    straightCount += 1
                else:
                    break

            images = np.array(images, dtype="float32")
            subParas = np.array(subParas, dtype="float32").reshape(-1, 1)
            labels = np.array(labels, dtype="float32").reshape(-1, 1)

            yield images, subParas, labels


    def __loadAllTrainingDataSet(self):
        dataSetDirNames = list(filter(lambda name: '2021' in name, os.listdir(f"{self.traceDataDirPath}/")))
        dataSetDirNames = np.random.permutation(dataSetDirNames)
        images = []
        subParas = []
        labels = []
        for dataSetDirName in dataSetDirNames:
            logPath = f"{self.traceDataDirPath}/{dataSetDirName}/driving_log.csv"
            print(f"start training {logPath} ...")
            logData = pd.read_csv(logPath, names=[
                'Center Image',
                'Left Image',
                'Right Image',
                'Steering Angle',
                'Throttle',
                'Break',
                'Speed'
            ])

            noneStraightDataCount = 0
            for index in logData.index[:-1]:
                logData.loc[index, 'Next Steering Angle'] = logData.loc[index+1, 'Steering Angle']
                if abs(logData.loc[index, 'Next Steering Angle']) <= 1e-2:
                    continue
                labels.append(logData.loc[index, 'Next Steering Angle'])
                # load images
                imageName = logData.loc[index, 'Center Image'].split('\\')[-1]
                imagePath = f"{self.traceDataDirPath}/{dataSetDirName}/IMG/{imageName}"
                image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # image = getCenterDeviationWithImage(image)
                images.append(image.astype("float32") / 255)

                # subPara = logData.loc[index, ['Steering Angle', 'Throttle', 'Speed']].values.ravel()
                # subPara[2] /= 30.5
                subPara = logData.loc[index, 'Steering Angle']
                subParas.append(subPara)
                noneStraightDataCount += 1

            # print(logData.groupby('Next Steering Angle').count())
            logData = logData.dropna()
            logData.to_csv(f"{self.traceDataDirPath}/{dataSetDirName}/new_log.csv")

            # add straight observations
            neededStraightAmount = noneStraightDataCount
            print(f"needed straight amount = {neededStraightAmount}")
            shuffledIndices = np.random.permutation(logData.index)
            straightCount = 0
            for index in shuffledIndices:
                if abs(logData.loc[index, 'Next Steering Angle']) > 1e-2:
                    continue
                labels.append(logData.loc[index, 'Next Steering Angle'])
                # load images
                imageName = logData.loc[index, 'Center Image'].split('\\')[-1]
                imagePath = f"{self.traceDataDirPath}/{dataSetDirName}/IMG/{imageName}"
                image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image.astype("float32") / 255)

                # subPara = logData.loc[index, ['Steering Angle', 'Throttle', 'Speed']].values.ravel()
                # subPara[2] /= 30.5
                subPara = logData.loc[index, 'Steering Angle']
                subParas.append(subPara)
                if straightCount < neededStraightAmount:
                    straightCount += 1
                else:
                    break

        images = np.array(images, dtype="float32")
        subParas = np.array(subParas, dtype="float32").reshape(-1, 1)
        labels = np.array(labels, dtype="float32").reshape(-1, 1)

        return images, subParas, labels


if __name__ == '__main__':
    deepModel = DeepModel()
    deepModel.run()
    # deepModel.showVedio()
