import os
import sys
import glob
import xml.etree.ElementTree as ET

import glob
import json
import os
import shutil
import operator
import sys
import argparse
import math

import numpy as np
import random #remove this later
random.seed(10)
from matplotlib import pyplot as plt

import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs

flags.DEFINE_string('classes', './data/cityscapes.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3_train_10.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_integer('num_classes', 9, 'number of classes in the model')

# Evaluation arguments
flags.DEFINE_string('output_files_path', './output', 'path for evaluation results')
flags.DEFINE_string('gt_path', "./dataset/txt_annotations/val", 'path to ground truth txt files')
flags.DEFINE_string('img_path', "./dataset/Images/val", 'path to be detected images')

flags.DEFINE_boolean('run_quiet', False, 'Run evaluation quiet or not')
flags.DEFINE_boolean('debug_mode', False, 'Use 6 images to test')
flags.DEFINE_boolean('draw_plot', True, 'Draw results plot or not')
flags.DEFINE_boolean('specific_iou_flagged', False, 'Use specific iou')

# if you set the following True, the txt data will be saved in gt_path
flags.DEFINE_boolean('generate_txt_from_xml', False, 'Generate ground truth txt files from xml')
flags.DEFINE_string('gt_xml_path', './dataset/annotations/val', 'path to xml gt data (Required if generate_txt_from_xml set True')

def generate_txt_annot(output_path, xml_path):
    """ convert the xml annotations to txt format """


    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # create VOC format files
    xml_list = glob.glob(os.path.join(xml_path, "*.xml"))
    if len(xml_list) == 0:
        print("Error: no .xml files found in ground-truth")
        sys.exit()
    for tmp_file in xml_list:
        file_name = os.path.basename(tmp_file)
        output_filename = file_name.replace(".xml",".txt")
    # 1. create new file (VOC format)
        with open(os.path.join(output_path, output_filename), "a") as new_f:
            root = ET.parse(tmp_file).getroot()
            for obj in root.findall('object'):
                obj_name = obj.find('name').text
                bndbox = obj.find('bndbox')
                left = bndbox.find('xmin').text
                top = bndbox.find('ymin').text
                right = bndbox.find('xmax').text
                bottom = bndbox.find('ymax').text
                new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
    print("Conversion completed!")


def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre


def log_average_miss_rate(prec, rec, num_images):
    """
        log-average miss rate:
            Calculated by averaging miss rates at 9 evenly spaced FPPI points
            between 10e-2 and 10e0, in log-space.
        output:
                lamr | log-average miss rate
                mr | miss rate
                fppi | false positives per image
        references:
            [1] Dollar, Piotr, et al. "Pedestrian Detection: An Evaluation of the
            State of the Art." Pattern Analysis and Machine Intelligence, IEEE
            Transactions on 34.4 (2012): 743 - 761.
    """

    # if there were no detections of that class
    if prec.size == 0:
        lamr = 0
        mr = 1
        fppi = 0
        return lamr, mr, fppi

    fppi = (1 - prec)
    mr = (1 - rec)

    fppi_tmp = np.insert(fppi, 0, -1.0)
    mr_tmp = np.insert(mr, 0, 1.0)

    # Use 9 evenly spaced reference points in log-space
    ref = np.logspace(-2.0, 0.0, num = 9)
    for i, ref_i in enumerate(ref):
        # np.where() will always find at least 1 index, since min(ref) = 0.01 and min(fppi_tmp) = -1.0
        j = np.where(fppi_tmp <= ref_i)[-1][-1]
        ref[i] = mr_tmp[j]

    # log(0) is undefined, so we use the np.maximum(1e-10, ref)
    lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))

    return lamr, mr, fppi


def is_float_between_0_and_1(value):
    """
    check if the number is a float between 0.0 and 1.0
    """
    try:
        val = float(value)
        if val > 0.0 and val < 1.0:
            return True
        else:
            return False
    except ValueError:
        return False


def error(msg):
    """
    throw error and exit
    """
    print(msg)
    sys.exit(0)


def file_lines_to_list(path):
    """
    Convert the lines of a file to a list
    """
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content


def draw_text_in_image(img, text, pos, color, line_width):
    """
    Draws text in image
    """
    font = cv2.FONT_HERSHEY_PLAIN
    fontScale = 1
    lineType = 1
    bottomLeftCornerOfText = pos
    cv2.putText(img, text,
            bottomLeftCornerOfText,
            font,
            fontScale,
            color,
            lineType)
    text_width, _ = cv2.getTextSize(text, font, fontScale, lineType)[0]
    return img, (line_width + text_width)


def adjust_axes(r, t, fig, axes):
    """
    Plot - adjust axes
    """
    # get text width for re-scaling
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    # get axis width in inches
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    # get axis limit
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1]*propotion])


def draw_plot_func(dictionary, n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color, true_p_bar): 
    """
    Draw plot using Matplotlib
    """
    # sort the dictionary by decreasing value, into a list of tuples
    sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))
    # unpacking the list of tuples into two lists
    sorted_keys, sorted_values = zip(*sorted_dic_by_value)
    # 
    if true_p_bar != "":
        """
        Special case to draw in:
            - green -> TP: True Positives (object detected and matches ground-truth)
            - red -> FP: False Positives (object detected but does not match ground-truth)
            - pink -> FN: False Negatives (object not detected but present in the ground-truth)
        """
        fp_sorted = []
        tp_sorted = []
        for key in sorted_keys:
            fp_sorted.append(dictionary[key] - true_p_bar[key])
            tp_sorted.append(true_p_bar[key])
        plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Positive')
        plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Positive', left=fp_sorted)
        # add legend
        plt.legend(loc='lower right')
        """
        Write number on side of bar
        """
        fig = plt.gcf() # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            fp_val = fp_sorted[i]
            tp_val = tp_sorted[i]
            fp_str_val = " " + str(fp_val)
            tp_str_val = fp_str_val + " " + str(tp_val)
            # trick to paint multicolor with offset:
            # first paint everything and then repaint the first number
            t = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
            plt.text(val, i, fp_str_val, color='crimson', va='center', fontweight='bold')
            if i == (len(sorted_values)-1): # largest bar
                adjust_axes(r, t, fig, axes)
    else:
        plt.barh(range(n_classes), sorted_values, color=plot_color)
        """
        Write number on side of bar
        """
        fig = plt.gcf() # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            str_val = " " + str(val) # add a space before
            if val < 1.0:
                str_val = " {0:.2f}".format(val)
            t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
            # re-set axes to show number inside the figure
            if i == (len(sorted_values)-1): # largest bar
                adjust_axes(r, t, fig, axes)
    # set window title
    fig.canvas.set_window_title(window_title)
    # write classes in y axis
    tick_font_size = 12
    plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)
    """
    Re-scale height accordingly
    """
    init_height = fig.get_figheight()
    # comput the matrix height in points and inches
    dpi = fig.dpi
    height_pt = n_classes * (tick_font_size * 1.4) # 1.4 (some spacing)
    height_in = height_pt / dpi
    # compute the required figure height 
    top_margin = 0.15 # in percentage of the figure height
    bottom_margin = 0.05 # in percentage of the figure height
    figure_height = height_in / (1 - top_margin - bottom_margin)
    # set new height
    if figure_height > init_height:
        fig.set_figheight(figure_height)

    # set plot title
    plt.title(plot_title, fontsize=14)
    # set axis titles
    # plt.xlabel('classes')
    plt.xlabel(x_label, fontsize='large')
    # adjust size of window
    fig.tight_layout()
    # save the plot
    fig.savefig(output_path)
    # show image
    if to_show:
        plt.show()
    # close the plot
    plt.close()


def save_gt_as_json(gt_path, output_path):
    if not os.path.exists(output_path): # if it doesn't exist already
        os.makedirs(output_path)
    # save each of the ground-truth files into a ".json" file.
    # Create a list of all the class names present in the ground-truth (gt_classes).

    # get a list with the ground-truth files
    ground_truth_files_list = glob.glob(gt_path + '/*.txt')
    if len(ground_truth_files_list) == 0:
        error("Error: No ground-truth files found!")
    ground_truth_files_list.sort()
    # dictionary with counter per class
    gt_counter_per_class = {}
    counter_images_per_class = {}

    gt_files = []
    all_bbox = [] # all bounding boxes
    # loops over ground truth files
    for txt_file in ground_truth_files_list:
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))

        lines_list = file_lines_to_list(txt_file)
        # create ground-truth dictionary
        bounding_boxes = []
        is_difficult = False
        already_seen_classes = []
        for line in lines_list:
            try:
                if "difficult" in line:
                        class_name, left, top, right, bottom, _difficult = line.split()
                        is_difficult = True
                        bbox = left + " " + top + " " + right + " " +bottom
                else:
                        class_name = ' '.join(line.split()[:-4])
                        bbox = ' '.join(line.split()[-4:])

            except ValueError:
                error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                error_msg += " Expected: <class_name> <left> <top> <right> <bottom> ['difficult']\n"
                error_msg += " Received: " + line
                error_msg += "\n\nIf you have a <class_name> with spaces between words you should remove them\n"
                error_msg += "by running the script \"remove_space.py\" or \"rename_class.py\" in the \"extra/\" folder."
                error(error_msg)

            if is_difficult:
                bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False, "difficult":True})
                is_difficult = False
            else:
                bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False})
                all_bbox.append({"file_id": file_id, "class_name": class_name, "bbox":bbox})
                # count that object
                if class_name in gt_counter_per_class:
                    gt_counter_per_class[class_name] += 1
                else:
                    # if class didn't exist yet
                    gt_counter_per_class[class_name] = 1

                if class_name not in already_seen_classes:
                    if class_name in counter_images_per_class:
                        counter_images_per_class[class_name] += 1
                    else:
                        # if class didn't exist yet
                        counter_images_per_class[class_name] = 1
                    already_seen_classes.append(class_name)


        # dump bounding_boxes into a ".json" file
        output_file = os.path.join(output_path, f"{file_id}.json")
        gt_files.append(output_file)
        with open(output_file, 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    # save also detections per class
    for class_name in gt_counter_per_class.keys():
        # search for class name
        detections = [item for item in all_bbox if item["class_name"]==class_name]
        output_file = os.path.join(output_path, f"{class_name}.json")
        with open(output_file, 'w') as outfile:
            json.dump(detections, outfile)
    

    return gt_counter_per_class, counter_images_per_class, ground_truth_files_list

  
def save_detections_as_json(img_path, gt_classes, output_path):
    img_output_path = os.path.join(output_path, "results_imgs")
    if not os.path.exists(output_path): # if it doesn't exist already
        os.makedirs(output_path)
        os.makedirs(img_output_path)

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights).expect_partial()
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')


    bounding_boxes = {k:[] for k in gt_classes}
    cities = glob.glob(f"{img_path}/*/") # subdirectory path in val folder
    t1 = int(time.time())
    count_files = 0
    for city in cities:
        img_list = glob.glob(f"{city}/*") # subdirectory path

        for img_path in img_list:
            file_id = img_path.split(img_path[-4:], 1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))

            img_raw = tf.image.decode_image(
                open(img_path, 'rb').read(), channels=3)

            img = tf.expand_dims(img_raw, 0)
            img = transform_images(img, FLAGS.size)
            boxes, scores, classes, nums = yolo(img)

            # parse detection results
            img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
            wh = np.flip(img.shape[0:2]) # to convert from x,y,h,w to xmin,ymin,xmax,ymax

            n_boxes = nums.numpy()[0]
            for i in range(n_boxes):
                class_name = class_names[int(classes[0][i])]
                if class_name not in gt_classes:
                    continue
                confidence = scores[0][i].numpy()
                confidence = float(confidence)
                x1y1 = np.array(boxes[0][i][:2] * wh).astype(np.int32)
                x2y2 = np.array(boxes[0][i][2:] * wh).astype(np.int32)
                bbox = list(x1y1) + list(x2y2) # convert to this format [xmin, ymin, xmax, ymax]
                bbox = " ".join(str(item) for item in bbox) # convert to this format: "xmin ymin xmax ymax" (as str)

                # append detection to dictionary
                bounding_boxes[class_name].append({"confidence":confidence, "file_id":file_id, "bbox":bbox})

            # save detect results as image
            img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
            saving_path = os.path.join(img_output_path, os.path.basename(os.path.normpath(img_path)))
            cv2.imwrite(saving_path, img)
            logging.info('output saved to: {}'.format(saving_path))
            count_files += 1
    
    t2 = int(time.time())
    logging.info('detection generation duration: {} second'.format(t2 - t1))

    # Saving json files
    for class_name, bb in bounding_boxes.items():
        new_temp_file = f"{output_path}/{class_name}_dr.json"
        with open(new_temp_file, 'w') as outfile:
            bb.sort(key=lambda x:float(x['confidence']), reverse=True)
            json.dump(bb, outfile)
    
    return bounding_boxes, count_files


def evaluate(_argv):
    output_files_path = FLAGS.output_files_path
    run_quiet = FLAGS.run_quiet
    debug_mode = FLAGS.debug_mode
    specific_iou_flagged = FLAGS.specific_iou_flagged
    draw_plot = FLAGS.draw_plot
    gt_path = FLAGS.gt_path
    img_path = FLAGS.img_path
    
    if FLAGS.generate_txt_from_xml:
        generate_txt_annot(FLAGS.gt_path, FLAGS.gt_xml_path)
   
    if os.path.exists(output_files_path): # if it exist already
        # reset the output directory
        shutil.rmtree(output_files_path)
    
    # create emtpy directories
    os.makedirs(output_files_path)
    os.makedirs(os.path.join(output_files_path, "classes"))

    if debug_mode:
        gt_path = "dataset/sample_txt/val"
        img_path = "dataset/sample_images/val"


    json_path = os.path.join(output_files_path, "json_files")
    gt_output_path = os.path.join(json_path, "gt")
    gt_counter_per_class, counter_images_per_class, ground_truth_files_list = save_gt_as_json(gt_path, gt_output_path)
    gt_classes = list(gt_counter_per_class.keys())
    gt_classes = sorted(gt_classes)
    n_classes = len(gt_classes)

    # dr = detection results
    dr_output_path = os.path.join(json_path, "dr")
    dr_bbox, n_detection_imgs = save_detections_as_json(img_path, gt_classes, dr_output_path)

    gt_files = glob.glob(os.path.join(gt_output_path, "*.json"))
    gt_data = dict()
    for gt_file in gt_files:
        file_id = gt_file.split(".json", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        gt_data[file_id] = json.load(open(gt_file))

    sum_AP = 0.0
    ap_dictionary = {}
    lamr_dictionary = {}

    # open file to store the output
    with open(output_files_path + "/output.txt", 'w') as output_file:
        output_file.write("# AP and precision/recall per class\n")
        count_true_positives = {}
        for class_index, class_name in enumerate(gt_classes):
            count_true_positives[class_name] = 0

            # Load detection-results of that class
            dr_file = f"{dr_output_path}/{class_name}_dr.json"
            dr_data = json.load(open(dr_file))

            # Assign detection-results to ground-truth objects
            nd = len(dr_data)
            tp = [0] * nd # creates an array of zeros of size nd
            fp = [0] * nd
            for idx, detection in enumerate(dr_data):
                file_id = detection["file_id"]

                # assign detection-results to ground truth object if any
                # open ground-truth with that file_id
                ground_truth_data = gt_data[file_id]
                ovmax = -1
                gt_match = -1
                # load detected object bounding-box
                bb = [ float(x) for x in detection["bbox"].split() ]
                for obj in ground_truth_data:
                    # look for a class_name match
                    if obj["class_name"] == class_name:
                        bbgt = [ float(x) for x in obj["bbox"].split() ]
                        bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
                        iw = bi[2] - bi[0] + 1
                        ih = bi[3] - bi[1] + 1
                        if iw > 0 and ih > 0:
                            # compute overlap (IoU) = area of intersection / area of union
                            ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                            + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                            ov = iw * ih / ua
                            if ov > ovmax:
                                ovmax = ov
                                gt_match = obj

                # assign detection as true positive/don't care/false positive
                # set minimum overlap
                min_overlap = MINOVERLAP
                if specific_iou_flagged:
                    if class_name in specific_iou_classes:
                        index = specific_iou_classes.index(class_name)
                        min_overlap = float(iou_list[index])
                if ovmax >= min_overlap:
                    if "difficult" not in gt_match:
                            if not bool(gt_match["used"]):
                                # true positive
                                tp[idx] = 1
                                gt_match["used"] = True
                                count_true_positives[class_name] += 1
                                # update the ".json" file
                                with open(gt_file, 'w') as f:
                                        f.write(json.dumps(ground_truth_data))
                            else:
                                # false positive (multiple detection)
                                fp[idx] = 1
                else:
                    # false positive
                    fp[idx] = 1
                    if ovmax > 0:
                        status = "INSUFFICIENT OVERLAP"

            #print(tp)
            # compute precision/recall
            cumsum = 0
            for idx, val in enumerate(fp):
                fp[idx] += cumsum
                cumsum += val
            cumsum = 0
            for idx, val in enumerate(tp):
                tp[idx] += cumsum
                cumsum += val
            #print(tp)
            rec = tp[:]
            for idx, val in enumerate(tp):
                rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
            #print(rec)
            prec = tp[:]
            for idx, val in enumerate(tp):
                prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
            #print(prec)

            ap, mrec, mprec = voc_ap(rec[:], prec[:])
            sum_AP += ap
            text = "{0:.2f}%".format(ap*100) + " = " + class_name + " AP " #class_name + " AP = {0:.2f}%".format(ap*100)

            ## Write to output.txt
            rounded_prec = [ '%.2f' % elem for elem in prec ]
            rounded_rec = [ '%.2f' % elem for elem in rec ]
            output_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall :" + str(rounded_rec) + "\n\n")
            if not run_quiet:
                print(text)
            ap_dictionary[class_name] = ap

            n_images = counter_images_per_class[class_name]
            lamr, mr, fppi = log_average_miss_rate(np.array(prec), np.array(rec), n_images)
            lamr_dictionary[class_name] = lamr


            ## Draw plot
            if draw_plot:
                plt.plot(rec, prec, '-o')
                # add a new penultimate point to the list (mrec[-2], 0.0)
                # since the last line segment (and respective area) do not affect the AP value
                area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
                area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]
                plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')
                # set window title
                fig = plt.gcf() # gcf - get current figure
                fig.canvas.set_window_title('AP ' + class_name)
                # set plot title
                plt.title('class: ' + text)
                #plt.suptitle('This is a somewhat long figure title', fontsize=16)
                # set axis titles
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                # optional - set axes
                axes = plt.gca() # gca - get current axes
                axes.set_xlim([0.0,1.0])
                axes.set_ylim([0.0,1.05]) # .05 to give some extra space
                # Alternative option -> wait for button to be pressed
                #while not plt.waitforbuttonpress(): pass # wait for key display
                # Alternative option -> normal display
                #plt.show()
                # save the plot
                fig.savefig(output_files_path + "/classes/" + class_name + ".png")
                plt.cla() # clear axes for next plot

        output_file.write("\n# mAP of all classes\n")
        mAP = sum_AP / n_classes
        text = "mAP = {0:.2f}%".format(mAP*100)
        output_file.write(text + "\n")
        print(text)

    ## Count total of detection-results
    # iterate through all the files
    det_counter_per_class = {k:len(v) for k, v in dr_bbox.items()}
    dr_classes = list(det_counter_per_class.keys())

    ## Plot the total number of occurences of each class in the ground-truth
    if draw_plot:
        window_title = "ground-truth-info"
        plot_title = "ground-truth\n"
        plot_title += "(" + str(len(ground_truth_files_list)) + " files and " + str(n_classes) + " classes)"
        x_label = "Number of objects per class"
        output_path = output_files_path + "/ground-truth-info.png"
        to_show = False
        plot_color = 'forestgreen'
        draw_plot_func(
            gt_counter_per_class,
            n_classes,
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            '',
            )

    ## Write number of ground-truth objects per class to results.txt
    with open(output_files_path + "/output.txt", 'a') as output_file:
        output_file.write("\n# Number of ground-truth objects per class\n")
        for class_name in sorted(gt_counter_per_class):
            output_file.write(class_name + ": " + str(gt_counter_per_class[class_name]) + "\n")

    ## Finish counting true positives
    for class_name in dr_classes:
        # if class exists in detection-result but not in ground-truth then there are no true positives in that class
        if class_name not in gt_classes:
            count_true_positives[class_name] = 0
    #print(count_true_positives)

    ## Plot the total number of occurences of each class in the "detection-results" folder
    if draw_plot:
        window_title = "detection-results-info"
        # Plot title
        plot_title = "detection-results\n"
        plot_title += "(" + str(n_detection_imgs) + " files and "
        count_non_zero_values_in_dictionary = sum(int(x) > 0 for x in list(det_counter_per_class.values()))
        plot_title += str(count_non_zero_values_in_dictionary) + " detected classes)"
        # end Plot title
        x_label = "Number of objects per class"
        output_path = output_files_path + "/detection-results-info.png"
        to_show = False
        plot_color = 'forestgreen'
        true_p_bar = count_true_positives
        draw_plot_func(
            det_counter_per_class,
            len(det_counter_per_class),
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            true_p_bar
            )

    ## Write number of detected objects per class to output.txt
    with open(output_files_path + "/output.txt", 'a') as output_file:
        output_file.write("\n# Number of detected objects per class\n")
        for class_name in sorted(dr_classes):
            n_det = det_counter_per_class[class_name]
            text = class_name + ": " + str(n_det)
            text += " (tp:" + str(count_true_positives[class_name]) + ""
            text += ", fp:" + str(n_det - count_true_positives[class_name]) + ")\n"
            output_file.write(text)

    ## Draw log-average miss rate plot (Show lamr of all classes in decreasing order)
    if draw_plot:
        window_title = "lamr"
        plot_title = "log-average miss rate"
        x_label = "log-average miss rate"
        output_path = output_files_path + "/lamr.png"
        to_show = False
        plot_color = 'royalblue'
        draw_plot_func(
            lamr_dictionary,
            n_classes,
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            ""
            )

    ## Draw mAP plot (Show AP's of all classes in decreasing order)
    if draw_plot:
        window_title = "mAP"
        plot_title = "mAP = {0:.2f}%".format(mAP*100)
        x_label = "Average Precision"
        output_path = output_files_path + "/mAP.png"
        to_show = True
        plot_color = 'royalblue'
        draw_plot_func(
            ap_dictionary,
            n_classes,
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            ""
            )


if __name__ == '__main__':
    MINOVERLAP = 0.5
    try:
        app.run(evaluate)
    except SystemExit:
        pass
