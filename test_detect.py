from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont
import glob
from timeit import default_timer as timer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint
checkpoint = 'checkpoint_ssd300.pth.tar'
checkpoint = torch.load(checkpoint)
start_epoch = checkpoint['epoch'] + 1
print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
model = checkpoint['model']
model = model.to(device)
model.eval()

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def detect_txt(original_image, min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    # Transform
    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.module.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    return det_boxes, det_labels, det_scores

if __name__ == '__main__':
    img_path = '/disk2/taekwang/VOC/VOC2007/JPEGImages/002137.jpg'
    #test_img_path_file = '/disk2/taekwang/fire_dataset/temp/ALL_object_image_list_test.txt'
    #txt_save_path = '/disk2/taekwang/fire_dataset/test_txt_save_path'
    test_img_path_file = '/disk2/taekwang/fire_dataset/ONLY_label_txt/only_object_image_list_test.txt'
    txt_save_path = '/disk2/taekwang/fire_dataset/ONLY_test_txt_save_path'
    cnt = 0
    f = open(test_img_path_file,'r')
    check_more_2 = []

    total_time = 0
    while True:

        line = f.readline()
        if not line: break

        line = line[:-1]
        #print(line)
        filename = line.split('/')[-1]
        #print(filename)
        original_image = Image.open(line, mode = 'r')
        original_image = original_image.convert('RGB')
        cnt +=1
        #print(cnt)
        prev_time = timer()
        det_boxes , det_labels, det_scores = detect_txt(original_image, min_score=0.2, max_overlap=0.5, top_k=200)
        curr_time = timer()
        exec_time = curr_time - prev_time
        total_time += exec_time
        #print(det_boxes , det_labels)
        #print(int(det_boxes[0][0]), int(det_labels[0][0]))
        if len(det_boxes) >=2:
            check_more_2.append(filename)
        f2 = open(os.path.join(txt_save_path,filename[:-4]+'.txt') , 'w')
        if int(det_labels[0][0]) == 0:
            f2.close()
        else:
            dir = 0
            for i_det in det_boxes:
                #print("########",i_det)
                print("######", det_scores)
                score = float(det_scores[0][0+dir])
                xmin = int(i_det[0])
                ymin = int(i_det[1])
                xmax = int(i_det[2])
                ymax = int(i_det[3])
                write_word = 'fire {0} {1} {2} {3} {4}\n'.format(score,xmin,ymin,xmax,ymax)
                f2.write(write_word)
                dir +=1
            f2.close()

    print(float(cnt/total_time))
    print(check_more_2)



    #original_image = Image.open(img_path, mode='r')
    #original_image = original_image.convert('RGB')
    #detect(original_image, min_score=0.2, max_overlap=0.5, top_k=200).save("/disk2/taekwang/VOC/out.jpg")
