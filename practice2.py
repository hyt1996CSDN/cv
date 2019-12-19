import re
import linecache
import os


FILEDIR = r"G:\photo\mtcnn_data\wider_face_split\wider_face_split\\"
file = open(FILEDIR + 'wider_face_train_bbx_gt.txt', 'r')


def count_lines(file):
    lines_quantity = 0
    while True:
        buffer = file.read(1024 * 8192)
        if not buffer:
            break
        lines_quantity += buffer.count('\n')
    file.close()
    return lines_quantity


lines = count_lines(file)

for i in range(lines):
    line = linecache.getline(FILEDIR + 'wider_face_train_bbx_gt.txt', i)
    if re.search('jpg', line):
        position = line.index('/')
        file_name = line[position + 1: -5]
        folder_name = line[:position]
        i += 1
        face_count = int(linecache.getline(FILEDIR + 'wider_face_train_bbx_gt.txt', i))
        for j in range(face_count):
            box_line = linecache.getline(FILEDIR + 'wider_face_train_bbx_gt.txt',i + j + 1)  # x1, y1, w, h, x1,y1 为人脸框左上角的坐标

            a=box_line.split()
            print(a)
            po_x1 = box_line.index(' ')
            x1 = box_line[:po_x1]
            po_y1 = box_line.index(' ', po_x1 + 1)
            y1 = box_line[po_x1:po_y1]
            po_w = box_line.index(' ', po_y1 + 1)
            w = box_line[po_y1:po_w]
            po_h = box_line.index(' ', po_w + 1)
            h = box_line[po_w:po_h]
            coordinates = x1 + y1 + w + h
            # print(coordinates)
            if not (os.path.exists(FILEDIR + "wider_face_train\\" + folder_name)):
                os.makedirs(FILEDIR + "wider_face_train\\" + folder_name)
            with open(FILEDIR + "wider_face_train\\" + folder_name + "\\" + file_name + ".txt", 'a') as f:
                f.write(coordinates + "\n")
            i += i + j + 1
