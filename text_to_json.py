import json
import os
import glob
from PIL import Image
from collections import OrderedDict

data_path = '/disk2/taekwang/fire_dataset'
image_path = data_path+'/video_frame'
#train_txt_file = os.path.join(data_path,'train_fire_18k.txt')
#test_txt_file = os.path.join(data_path,'train_fire_2k_test.txt')
train_txt_file = os.path.join(data_path,'train_list.txt')
test_txt_file = os.path.join(data_path,'test_list.txt')

# 파일 개수 확인
test_cnt = 0
train_cnt = 0
f = open(test_txt_file, 'r')
while True:
    line = f.readline()
    if not line: break
    test_cnt +=1
f.close()

f = open(train_txt_file, 'r')
while True:
    line = f.readline()
    if not line: break
    train_cnt +=1
f.close()
print(test_cnt, train_cnt, test_cnt+train_cnt)


def parse_label():
    f = open(test_txt_file, 'r')
    f2 = open(os.path.join(data_path, 'temp/ALL_object_test_list_test.txt'), 'w')
    f3 = open(os.path.join(data_path, 'temp/ALL_object_image_list_test.txt'), 'w')
    f4 = open(os.path.join(data_path, 'temp/ALL_object_label_test.txt'), 'w')

    while True:
        line = f.readline()
        line = line.rstrip()
        if not line: break
        f2.write(line)
        f2.write('\n')
        f3.write(line.split()[0])
        f3.write('\n')
        f4.write(line.replace(line.split()[0],"").lstrip())
        f4.write('\n')
    f.close()
    f2.close()
    f3.close()
    f4.close()

#parse_label()

def parse_test_label():
    #f = open(os.path.join(data_path, 'temp/ALL_object_test_list_test.txt'), 'r')
    #savepath = '/disk2/taekwang/fire_dataset/test_txt_label_path'
    f = open(os.path.join(data_path, 'ONLY_label_txt/only_object_test_list_test.txt'), 'r')
    savepath = '/disk2/taekwang/fire_dataset/ONLY_test_txt_label_path'
    while True:
        line = f.readline()
        if not line: break

        path = line.split()[0]
        labels = line.split()[1:]

        #print(labels)
        file_name = path.split('/')[-1][:-4] + '.txt'
        print(labels)
        f2 = open(os.path.join(savepath, file_name), 'w')
        for idx in labels:
            if len(idx) == 0:
                f2.close()
            else:
                idx = idx.split(',')
                print("#############",idx)
                write_word = 'fire {0} {1} {2} {3}\n'.format(int(idx[0]),int(idx[1]),int(idx[2]),int(idx[3]))
                f2.write(write_word)
        f2.close()
    f.close()



#parse_test_label()
def parse_only_label():
    f = open(test_txt_file, 'r')
    f2 = open(os.path.join(data_path,'only_object_test_list_test.txt'),'w')
    f3 = open(os.path.join(data_path,'only_object_image_list_test.txt'),'w')
    f4 = open(os.path.join(data_path,'only_object_label_test.txt'),'w')
    while True:
        line = f.readline()
        line = line.rstrip()
        if not line: break
        #print(line[-1])
        if line[-1]=='1':
            f2.write(line)
            f2.write('\n')
            f3.write(line.split()[0])
            f3.write('\n')
            f4.write(line.replace(line.split()[0],"").lstrip())
            f4.write('\n')
    f.close()
    f2.close()
    f3.close()
    f4.close()
#parse_only_label()


def make_folder_label():
    cnt = 0
    f = open(os.path.join(data_path,'train_list.txt'),'r')
    fire_save_path = '/disk2/taekwang/fire_dataset/DataFolder/fire'
    normal_save_path = '/disk2/taekwang/fire_dataset/DataFolder/background'
    while True:
        cnt +=1
        line = f.readline().rstrip()
        if not line: break
        file_path = line.split()[0]
        fire_or_not = line.split()[-1][-1]
        print(cnt)
        img = Image.open(file_path)
        file_name = file_path.split('/')[-1]

        if fire_or_not=='1': # fire
            img.save(os.path.join(fire_save_path,file_name),"JPEG")
        else:
            img.save(os.path.join(normal_save_path, file_name), "JPEG")

make_folder_label()

#make json
def make_label_json():
    cnt = 0
    f = open(os.path.join(data_path,'only_object_label_test.txt'),'r')
    all_list = []
    while True:
        file_data = OrderedDict()

        file_data["boxes"] = []
        file_data["labels"] = []
        file_data["difficulties"] = []
        line = f.readline()
        object = line.split()
        if not line: break
        for i in range(len(object)):
            loc = object[i].split(',')[:-1]
            label = object[i].split(',')[-1][0]
            #print(loc,label)
            #file_data["boxes"].append(loc)
            file_data["boxes"].append(list(map(int,loc)))
            file_data["labels"].append(int(label))
            file_data["difficulties"].append(int(0))
        all_list.append(file_data)
        cnt += 1
        print("check", cnt)
    with open(os.path.join(data_path,'TEST_objects.json'), 'w') as j:
        json.dump(all_list, j)

    f.close()

def make_image_list_json():
    cnt = 0
    f = open(os.path.join(data_path, 'only_object_image_list_test.txt'), 'r')
    all_list = []
    while True:
        line = f.readline()
        if not line: break
        all_list.append(line[:-1])
        cnt +=1
        #print("check", cnt)
    with open(os.path.join(data_path, 'TEST_images.json'), 'w') as j:
        json.dump(all_list, j)



#make_label_json()
#make_image_list_json()