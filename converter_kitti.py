from scipy import misc
from glob import glob
import json, sys, os
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
from PIL import ImageEnhance, Image

img_width = 1920#1080 #640
img_height = 1072#1920 #360
root_folder_images = '/home/siiva/RUDY/SIIVA/GoalCam/annotations/NBA_CBA_CUBA/190918/002/original_data/frames' #'/media/siiva/C0E28D29E28D252E/goalCAM/BASKETBALL/RUCKER/frames_rucker3'

rescale_x = 1.0 * img_width / float(1920)
rescale_y = 1.0 * img_height / float(1072)

ignore_class = -1# dont care
coco_anno_path = '/home/siiva/RUDY/SIIVA/GoalCam/annotations/NBA_CBA_CUBA/190918/002' #'/home/siiva/RUDY/SIIVA/GoalCam/annotations/rucker/basketball_rucker3'

annotation_files = [f for f in listdir(coco_anno_path) if isfile(join(coco_anno_path, f))]

print("%d annotations files found in folder %s" % (len(annotation_files), annotation_files))

# loading all the files and annotations
for anno_file_id in range(len(annotation_files)):
    jsonfilename = annotation_files[anno_file_id]
    annotype = jsonfilename.split('_')[0]

    if annotype == '01ball':
        samples_class = 1
    elif annotype == '02basket':
        samples_class = 2
    # elif annotype == '03player':
    #     samples_class = 3

    with open(coco_anno_path+'/'+annotation_files[anno_file_id]) as json_data:
        # get Json annotations
        print("Using annotation: %s" % (coco_anno_path+'/'+annotation_files[anno_file_id]))
        data = json.load(json_data)

        # locations where BBoxes are going to be saved
        bbox_x1 = []
        bbox_y1 = []
        bbox_x2 = []
        bbox_y2 = []
        classes_id = []

        bbox_x1_fake = []
        bbox_y1_fake = []
        bbox_x2_fake = []
        bbox_y2_fake = []
        classes_id_fake = []

        images_path = []

        # looping for all the files
        last_save = None
        annotations_counter = 0
        for ann_id in range(len(data['annotations'])):

            if ignore_class == data['annotations'][ann_id]['category_id']:
                print("ignoring this class ... %d" % data['annotations'][ann_id]['category_id'])
                continue

            # getting bounding boxes
            x1 = data['annotations'][ann_id]['bbox'][0] * rescale_x
            y1 = data['annotations'][ann_id]['bbox'][1] * rescale_y
            x2 = (data['annotations'][ann_id]['bbox'][0] + data['annotations'][ann_id]['bbox'][2]) * rescale_x
            y2 = (data['annotations'][ann_id]['bbox'][1] + data['annotations'][ann_id]['bbox'][3]) * rescale_y

            # check all annotations are valid
            if x1 < 0 or y1 < 0 or x2 >= img_width or y1 >= img_height or x1 >= x2-5 or y1 >= y2-5:
                # this BBox is illegal
                print(x1, y1, x2, y2)
                print("ERROR ---> This annotation is illegal")
                continue

            # get annotations and images
            img_id = data['annotations'][ann_id]['image_id']

            if len(data['images']) > img_id:
                image_path = data['images'][img_id]['file_name']
            else:
                print("Image ID not found: %d requested of %d available" % (img_id, len(data['images'])))
                continue

            annotations_counter += 1

            images_path.append(image_path)
            classes_id.append(samples_class)#data['annotations'][ann_id]['category_id']

            # real annotation
            bbox_x1.append(x1)# left top x
            bbox_y1.append(y1)# left top y
            bbox_x2.append(x2)# right bottom x
            bbox_y2.append(y2)# right bottom y

            print("Image: %s\n "
                  "BBox extracted was (REAL): [%d, %d, %d %d] [w: %d, h: %d] [class: %d]"%
                  (images_path[len(images_path)-1], bbox_x1[len(bbox_x1)-1], bbox_y1[len(bbox_y1)-1],
                   bbox_x2[len(bbox_x2)-1], bbox_y2[len(bbox_y2)-1],
                   data['annotations'][ann_id]['bbox'][2], data['annotations'][ann_id]['bbox'][3],
                   classes_id[len(classes_id)-1]))

        print("Data conversions is summarized as follows:\n"
              "N_images: %d\n"
              "N_annotations: [%d/%d]\n"
              "Resolutions: %dx%d\n" % (len(data['images']), len(bbox_x1), annotations_counter, rescale_y*img_height,
                                        rescale_x*img_width))

    # =====================================================================================================================
    '''
        format is as follows:

        1 type Describes the type of object: '00normal', '01dont_care', '02black', '03sour', '04shell',
        '05fungus', '06insect', '07unknow'

        1 truncated Float from 0 (non-truncated) to 1 (truncated), where truncated refers to the object leaving image boundaries

        1 occluded Integer (0,1,2,3) indicating occlusion state:

        0 = fully visible, 1 = partly occluded

        4 bbox 2D bounding box of object in the image (0-based index): contains left, top, right, bottom pixel coordinates

        3 dimensions 3D object dimensions: height, width, length (in meters)

        3 location 3D object location x,y,z in camera coordinates (in meters)

        1 rotation_y Rotation ry around Y-axis in camera coordinates [-pi..pi]

        1 score Only for results: Float, indicating confidence in detection, needed for p/r curves, higher is better.

        2 = largely occluded, 3 = unknown

        1 alpha Observation angle of object, ranging [-pi..pi]
    '''


    image_files_path = '/home/siiva/RUDY/SIIVA/GoalCam/annotations/NBA_CBA_CUBA/190918/002/training/image_2'
    labels_files_path = '/home/siiva/RUDY/SIIVA/GoalCam/annotations/NBA_CBA_CUBA/190918/002/training/label_2'
    training_files_path = '/home/siiva/RUDY/SIIVA/GoalCam/annotations/NBA_CBA_CUBA/190918/002/train.txt'
    debug_images_path = '/home/siiva/RUDY/SIIVA/GoalCam/annotations/NBA_CBA_CUBA/190918/002/debug_images'

    # opening training dataset
    training_set = open(training_files_path, 'a')

    class_name = ['dummy', '01ball', '02basket']# , '03player']

    last_save = None
    truncated = 0.0
    ocluded = 0.0
    fully_visible = 0.0
    large_ocluded = 0.0
    alpha = 0.0
    general = 0.0


    n_fake_sample_id = 0
    for anno_id in range(annotations_counter):
        file_name_combined = images_path[anno_id].split('/')
        file_name_single_ext = file_name_combined[len(file_name_combined)-1]
        file_name = file_name_single_ext.split('.')[0]

        annotation_file = open(labels_files_path+'/'+file_name+'.txt', 'a')

        print("USING ANNOTATION FILE (%d): %s" % (anno_id, (labels_files_path+'/'+file_name+'.txt')))
        print(training_set)

        # writing to file REAL Bboxes
        annotation_file.write(
            class_name[classes_id[anno_id]] + ' ' +
            str(truncated) + ' ' +
            str(ocluded) + ' ' +
            str(fully_visible) + ' ' +
            str(bbox_x1[anno_id]) + ' ' +
            str(bbox_y1[anno_id]) + ' ' +
            str(bbox_x2[anno_id]) + ' ' +
            str(bbox_y2[anno_id]) + ' ' +
            str(general) + ' ' +
            str(general) + ' ' +
            str(general) + ' ' +
            str(general) + ' ' +
            str(general) + ' ' +
            str(general) + ' ' +
            str(general)
        )
        annotation_file.write('\n')

        # flushing everything
        annotation_file.flush()
        annotation_file.close()

        if last_save is not images_path[anno_id]:
            filename2save = image_files_path+'/'+file_name+'.bmp'
            if os.path.exists(filename2save):
                pass
            else:
                last_save = root_folder_images+'/'+images_path[anno_id]

                if not os.path.exists(last_save):
                    continue

                img = misc.imread(last_save, mode='RGB')
                img = misc.imresize(img, [img_height, img_width])
                misc.imsave(image_files_path+'/'+file_name+'.jpg', img)
                training_set.write(file_name+'\n')
                training_set.flush()
                print("annotation on file %s, done" % last_save)

        #
        # ===========================================================
        # Color Image
        # ===========================================================
        #
        for idx in range(3):
            file_name_combined = images_path[anno_id].split('/')
            file_name_single_ext = file_name_combined[len(file_name_combined)-1]
            file_name = file_name_single_ext.split('.')[0] + ("_color_%d"%idx)

            annotation_file = open(labels_files_path+'/'+file_name+'.txt', 'a')

            # writing to file REAL Bboxes
            annotation_file.write(
                class_name[classes_id[anno_id]] + ' ' +
                str(truncated) + ' ' +
                str(ocluded) + ' ' +
                str(fully_visible) + ' ' +
                str(bbox_x1[anno_id]) + ' ' +
                str(bbox_y1[anno_id]) + ' ' +
                str(bbox_x2[anno_id]) + ' ' +
                str(bbox_y2[anno_id]) + ' ' +
                str(general) + ' ' +
                str(general) + ' ' +
                str(general) + ' ' +
                str(general) + ' ' +
                str(general) + ' ' +
                str(general) + ' ' +
                str(general)
            )
            annotation_file.write('\n')

            # flushing everything
            annotation_file.flush()
            annotation_file.close()

            if last_save is not images_path[anno_id]:
                filename2save = image_files_path+'/'+file_name+'.bmp'
                if os.path.exists(filename2save):
                    pass
                else:
                    last_save = root_folder_images+'/'+images_path[anno_id]

                    if not os.path.exists(last_save):
                        continue

                    img = misc.imread(last_save, mode='RGB')
                    img = misc.imresize(img, [img_height, img_width])
                    pil_image = Image.fromarray(img, "RGB")
                    colorer = ImageEnhance.Color(pil_image)
                    factor = 0.3 * (idx+1)
                    colored_image = colorer.enhance(factor)
                    img = np.asarray(colored_image)

                    misc.imsave(image_files_path+'/'+file_name+'.jpg', img)
                    training_set.write(file_name+'\n')
                    training_set.flush()
                    print("annotation on file %s, done" % last_save)

        #
        # ===========================================================
        # Brightness Image
        # ===========================================================
        #
        for idx in range(3):
            file_name_combined = images_path[anno_id].split('/')
            file_name_single_ext = file_name_combined[len(file_name_combined)-1]
            file_name = file_name_single_ext.split('.')[0] + ("_brightness_%d"%idx)

            annotation_file = open(labels_files_path+'/'+file_name+'.txt', 'a')

            # writing to file REAL Bboxes
            annotation_file.write(
	            class_name[classes_id[anno_id]] + ' ' +
	            str(truncated) + ' ' +
	            str(ocluded) + ' ' +
	            str(fully_visible) + ' ' +
	            str(bbox_x1[anno_id]) + ' ' +
	            str(bbox_y1[anno_id]) + ' ' +
	            str(bbox_x2[anno_id]) + ' ' +
	            str(bbox_y2[anno_id]) + ' ' +
	            str(general) + ' ' +
	            str(general) + ' ' +
	            str(general) + ' ' +
	            str(general) + ' ' +
	            str(general) + ' ' +
	            str(general) + ' ' +
	            str(general)
	        )
            annotation_file.write('\n')

            # flushing everything
            annotation_file.flush()
            annotation_file.close()

            if last_save is not images_path[anno_id]:
                filename2save = image_files_path+'/'+file_name+'.bmp'

                if os.path.exists(filename2save):
                    pass
                else:
                    last_save = root_folder_images+'/'+images_path[anno_id]

                    if not os.path.exists(last_save):
                        continue

                    img = misc.imread(last_save, mode='RGB')
                    img = misc.imresize(img, [img_height, img_width])
                    pil_image = Image.fromarray(img, "RGB")
                    brighter = ImageEnhance.Brightness(pil_image)
                    factor = 0.3 * (idx+1)
                    bright_image = brighter.enhance(factor)
                    img = np.asarray(bright_image)

                    misc.imsave(image_files_path+'/'+file_name+'.jpg', img)
                    training_set.write(file_name+'\n')
                    training_set.flush()
                    print("annotation on file %s, done" % last_save)

        #
        # ===========================================================
        # Contrast Image
        # ===========================================================
        #
        for idx in range(3):
            file_name_combined = images_path[anno_id].split('/')
            file_name_single_ext = file_name_combined[len(file_name_combined)-1]
            file_name = file_name_single_ext.split('.')[0] + ("_contrast_%d"%idx)

            annotation_file = open(labels_files_path+'/'+file_name+'.txt', 'a')

            # writing to file REAL Bboxes
            annotation_file.write(
	            class_name[classes_id[anno_id]] + ' ' +
	            str(truncated) + ' ' +
	            str(ocluded) + ' ' +
	            str(fully_visible) + ' ' +
	            str(bbox_x1[anno_id]) + ' ' +
	            str(bbox_y1[anno_id]) + ' ' +
	            str(bbox_x2[anno_id]) + ' ' +
	            str(bbox_y2[anno_id]) + ' ' +
	            str(general) + ' ' +
	            str(general) + ' ' +
	            str(general) + ' ' +
	            str(general) + ' ' +
	            str(general) + ' ' +
	            str(general) + ' ' +
	            str(general)
	        )
            annotation_file.write('\n')

            # flushing everything
            annotation_file.flush()
            annotation_file.close()

            if last_save is not images_path[anno_id]:
                filename2save = image_files_path+'/'+file_name+'.bmp'

                if os.path.exists(filename2save):
                    pass
                else:
                    last_save = root_folder_images+'/'+images_path[anno_id]

                    if not os.path.exists(last_save):
                        continue

                    img = misc.imread(last_save, mode='RGB')
                    img = misc.imresize(img, [img_height, img_width])
                    pil_image = Image.fromarray(img, "RGB")
                    contraster = ImageEnhance.Contrast(pil_image)
                    factor = 0.3 * (idx+1)
                    contrasted_image = contraster.enhance(factor)
                    img = np.asarray(contrasted_image)

                    misc.imsave(image_files_path+'/'+file_name+'.jpg', img)
                    training_set.write(file_name+'\n')
                    training_set.flush()
                    print("annotation on file %s, done" % last_save)

        #
        # ===========================================================
        # BLurring Image
        # ===========================================================
        #
        for idx in range(5):
            file_name_combined = images_path[anno_id].split('/')
            file_name_single_ext = file_name_combined[len(file_name_combined)-1]
            file_name = file_name_single_ext.split('.')[0] + ("_sharp_%d"%idx)

            annotation_file = open(labels_files_path+'/'+file_name+'.txt', 'a')

            # writing to file REAL Bboxes
            annotation_file.write(
	            class_name[classes_id[anno_id]] + ' ' +
	            str(truncated) + ' ' +
	            str(ocluded) + ' ' +
	            str(fully_visible) + ' ' +
	            str(bbox_x1[anno_id]) + ' ' +
	            str(bbox_y1[anno_id]) + ' ' +
	            str(bbox_x2[anno_id]) + ' ' +
	            str(bbox_y2[anno_id]) + ' ' +
	            str(general) + ' ' +
	            str(general) + ' ' +
	            str(general) + ' ' +
	            str(general) + ' ' +
	            str(general) + ' ' +
	            str(general) + ' ' +
	            str(general)
	        )
            annotation_file.write('\n')

            # flushing everything
            annotation_file.flush()
            annotation_file.close()

            if last_save is not images_path[anno_id]:
                filename2save = image_files_path+'/'+file_name+'.bmp'

                if os.path.exists(filename2save):
                    pass
                else:
                    last_save = root_folder_images+'/'+images_path[anno_id]

                    if not os.path.exists(last_save):
                        continue

                    img = misc.imread(last_save, mode='RGB')
                    img = misc.imresize(img, [img_height, img_width])
                    pil_image = Image.fromarray(img, "RGB")
                    blurrer = ImageEnhance.Sharpness(pil_image)
                    factor = 0.4 * (idx+1)
                    blurred_image = blurrer.enhance(factor)
                    img = np.asarray(blurred_image)

                    misc.imsave(image_files_path+'/'+file_name+'.jpg', img)
                    training_set.write(file_name+'\n')
                    training_set.flush()
                    print("annotation on file %s, done" % last_save)


    training_set.close()

    if True:
        image_files = [f for f in listdir(image_files_path) if isfile(join(image_files_path, f))]

        for im_id in range(len(image_files)):
            image_basename = image_files[im_id].split('.')[0]
            annotation_image = labels_files_path + '/' + image_basename + '.txt'

            global_counter_here = 0

            with open(annotation_image) as f:
                content = f.readlines()
                content = [x.strip() for x in content]

                global_counter_here += 1
                if global_counter_here == 1:
                    img = misc.imread(image_files_path+'/'+image_files[im_id], mode='RGB')
                    img = misc.imresize(img, [img_height, img_width])
                elif global_counter_here > 1:
                    img = misc.imread(debug_images_path+'/'+image_basename+'.jpg', mode='RGB')
                    img = misc.imresize(img, [img_height, img_width])
                for ann_idx in range(len(content)):
                    # plus the real one
                    content_intern = content[ann_idx].split(' ')
                    print(content_intern)
                    class_name = content_intern[0]
                    content_intern = np.array(content_intern[1:]).astype(np.float)
                    print(class_name, content_intern, content_intern.shape)

                    cv2.rectangle(img, (int(content_intern[3]), int(content_intern[4])),
                        (int(content_intern[5]), int(content_intern[6])), (51,255,255), 2)

                    cv2.putText(img, class_name, (int(content_intern[3]+20), int(content_intern[4]+20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (51,255,51), 2)

                    #closing image
                    misc.imsave(debug_images_path+'/'+image_basename+'.jpg', img)

    # sys.exit(0)
