import os
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import matplotlib.pyplot as plt

img_paths = "/media/siiva/C0E28D29E28D252E/goalCAM/BASKETBALL/KITTI/training2/image_2/"
lbl_paths = "/media/siiva/C0E28D29E28D252E/goalCAM/BASKETBALL/KITTI/training2/label_2/"

with open('/media/siiva/C0E28D29E28D252E/goalCAM/BASKETBALL/KITTI/ImageSets/train.txt', 'r') as f:
    for line in f:
        print(line.strip())
        img = Image.open(img_paths + line.strip() + '.jpg')

        img = np.array(img)
        img_resize = cv2.resize(img, (640, 360))
        img_size = np.asarray(img.shape)[0:2]
        print(img_size)
        rescale_y = 1.0 * 360 / img_size[0]
        rescale_x = 1.0 * 640 / img_size[1]

        pos = 0
        with open(lbl_paths + line.strip() + '.txt', 'r') as l:
            for label in l:
                label = label.strip().split(' ')
                label[0] = '01ball'
                bb_0 = float(label[4])
                bb_2 = float(label[6])
                bb_1 = float(label[5])
                bb_3 = float(label[7])

                bb_0 *= rescale_x
                bb_2 *= rescale_x
                bb_1 *= rescale_y
                bb_3 *= rescale_y

                if bb_0 > 0 and bb_1 > 0 and bb_2 > 0 and bb_3 > 0:
                    l1 = open(
                        '/media/siiva/C0E28D29E28D252E/goalCAM/BASKETBALL/KITTI/training2/label_2/' + line.strip() + '.txt',
                        'w')
                    pos = 1

                    l1.write(
                        str(label[0]) + ' ' + str(label[1]) + ' ' + str(label[2]) + ' ' + str(label[3]) + ' ' + str(
                            bb_0) + ' ' + str(bb_1) + ' ' + str(bb_2) + ' ' + str(bb_3) + ' ' + str(
                            label[8]) + ' ' + str(label[9]) + ' ' + str(label[10]) + ' ' + str(label[11]) + ' ' + str(
                            label[12]) + ' ' + str(label[13]) + ' ' + str(label[14]))
                    l1.write('\n')
                    l1.close()

                    # ===========================================================
                    # Color Image
                    # ===========================================================
                    #
                    for idx in range(3):
                        l1 = open(
                            '/media/siiva/C0E28D29E28D252E/goalCAM/BASKETBALL/KITTI/training2/label_2/' + 'color_' + str(idx)+ '_' + line.strip() + '.txt',
                            'w')
                        l1.write(
                            str(label[0]) + ' ' + str(label[1]) + ' ' + str(label[2]) + ' ' + str(label[3]) + ' ' + str(
                                bb_0) + ' ' + str(bb_1) + ' ' + str(bb_2) + ' ' + str(bb_3) + ' ' + str(
                                label[8]) + ' ' + str(label[9]) + ' ' + str(label[10]) + ' ' + str(
                                label[11]) + ' ' + str(
                                label[12]) + ' ' + str(label[13]) + ' ' + str(label[14]))
                        l1.write('\n')
                        l1.close()

                    # ===========================================================
                    # Brightness Image
                    # ===========================================================
                    #
                    for idx in range(3):
                        l1 = open(
                            '/media/siiva/C0E28D29E28D252E/goalCAM/BASKETBALL/KITTI/training2/label_2/' + 'brightness_' + str(idx)+ '_' + line.strip() + '.txt',
                            'w')
                        l1.write(
                            str(label[0]) + ' ' + str(label[1]) + ' ' + str(label[2]) + ' ' + str(label[3]) + ' ' + str(
                                bb_0) + ' ' + str(bb_1) + ' ' + str(bb_2) + ' ' + str(bb_3) + ' ' + str(
                                label[8]) + ' ' + str(label[9]) + ' ' + str(label[10]) + ' ' + str(
                                label[11]) + ' ' + str(
                                label[12]) + ' ' + str(label[13]) + ' ' + str(label[14]))
                        l1.write('\n')
                        l1.close()

                    # ===========================================================
                    # Contrast Image
                    # ===========================================================
                    #
                    for idx in range(3):
                        l1 = open(
                            '/media/siiva/C0E28D29E28D252E/goalCAM/BASKETBALL/KITTI/training2/label_2/' + 'contrast_' + str(idx)+ '_' + line.strip() + '.txt',
                            'w')
                        l1.write(
                            str(label[0]) + ' ' + str(label[1]) + ' ' + str(label[2]) + ' ' + str(label[3]) + ' ' + str(
                                bb_0) + ' ' + str(bb_1) + ' ' + str(bb_2) + ' ' + str(bb_3) + ' ' + str(
                                label[8]) + ' ' + str(label[9]) + ' ' + str(label[10]) + ' ' + str(
                                label[11]) + ' ' + str(
                                label[12]) + ' ' + str(label[13]) + ' ' + str(label[14]))
                        l1.write('\n')
                        l1.close()

                    # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == =
                    # Contrast Image
                    # ===========================================================
                    #
                    for idx in range(3):
                        l1 = open(
                            '/media/siiva/C0E28D29E28D252E/goalCAM/BASKETBALL/KITTI/training2/label_2/' + 'blur_' + str(idx)+ '_' + line.strip() + '.txt',
                            'w')
                        l1.write(
                            str(label[0]) + ' ' + str(label[1]) + ' ' + str(label[2]) + ' ' + str(label[3]) + ' ' + str(
                                bb_0) + ' ' + str(bb_1) + ' ' + str(bb_2) + ' ' + str(bb_3) + ' ' + str(
                                label[8]) + ' ' + str(label[9]) + ' ' + str(label[10]) + ' ' + str(
                                label[11]) + ' ' + str(
                                label[12]) + ' ' + str(label[13]) + ' ' + str(label[14]))
                        l1.write('\n')
                        l1.close()

        # cv2.rectangle(img, (int(float(label[4])), int(float(label[5]))), (int(float(label[6])), int(float(label[7]))), (0, 255, 0), 2)
        # plt.imshow(img[:,:,::-1])
        # plt.show()
        # cv2.rectangle(img_resize, (int(bb_0), int(bb_1)), (int(bb_2), int(bb_3)), (0, 255, 0), 2)
        # plt.imshow(img_resize[:,:,::-1])
        # plt.show()
        if pos == 1:
            cv2.imwrite('/media/siiva/C0E28D29E28D252E/goalCAM/BASKETBALL/KITTI/training2/image_2/' + line.strip() + '.jpg',
                        img_resize[:, :, ::-1])
            for idx in range(3):
                pil_image = Image.fromarray(img_resize, "RGB")
                colorer = ImageEnhance.Color(pil_image)
                factor = 0.3 * (idx + 1)
                colored_image = colorer.enhance(factor)
                img = np.asarray(colored_image)

                cv2.imwrite('/media/siiva/C0E28D29E28D252E/goalCAM/BASKETBALL/KITTI/training2/image_2/' + 'color_' + str(idx)+ '_' + line.strip() + '.jpg',
                            img[:, :, ::-1])

                # cv2.rectangle(img_resize, (int(bb_0), int(bb_1)), (int(bb_2), int(bb_3)), (0, 255, 0), 2)
                # plt.imshow(img_resize)
                # plt.show()

            for idx in range(3):
                pil_image = Image.fromarray(img_resize, "RGB")
                brighter = ImageEnhance.Brightness(pil_image)
                factor = 0.3 * (idx + 1)
                bright_image = brighter.enhance(factor)
                img = np.asarray(bright_image)

                cv2.imwrite('/media/siiva/C0E28D29E28D252E/goalCAM/BASKETBALL/KITTI/training2/image_2/' + 'brightness_' + str(idx)+ '_' + line.strip() + '.jpg',
                            img[:, :, ::-1])


            for idx in range(3):
                pil_image = Image.fromarray(img_resize, "RGB")
                contraster = ImageEnhance.Contrast(pil_image)
                factor = 0.3 * (idx + 1)
                contrasted_image = contraster.enhance(factor)
                img = np.asarray(contrasted_image)

                cv2.imwrite('/media/siiva/C0E28D29E28D252E/goalCAM/BASKETBALL/KITTI/training2/image_2/' + 'contrast_' + str(idx)+ '_' + line.strip() + '.jpg',
                            img[:, :, ::-1])


            for idx in range(3):
                pil_image = Image.fromarray(img_resize, "RGB")
                blurrer = ImageEnhance.Sharpness(pil_image)
                factor = 0.4 * (idx + 1)
                blurred_image = blurrer.enhance(factor)
                img = np.asarray(blurred_image)

                cv2.imwrite('/media/siiva/C0E28D29E28D252E/goalCAM/BASKETBALL/KITTI/training2/image_2/' + 'blur_' + str(idx)+ '_' + line.strip() + '.jpg',
                            img[:, :, ::-1])


