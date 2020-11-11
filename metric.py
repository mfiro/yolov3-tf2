import sys
import os
import glob
import xml.etree.ElementTree as ET
# This file is originally from https://github.com/Cartucho/mAP

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

def main():
    output_path = "./dataset/txt_annotations/val/"
    xml_path = "./dataset/annotations/val/"
    generate_txt_annot(output_path, xml_path)


if __name__ == "__main__":
    main()