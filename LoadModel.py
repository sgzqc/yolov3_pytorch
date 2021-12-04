import numpy as np
import cv2
import torch
from yolov3 import YOLOV3
from utils import image_preprocess,img_loader,postprocess_boxes,nms,draw_bbox


def read_param_from_file(yolo_ckpt,model):
    wf = open(yolo_ckpt, 'rb')
    major, minor, vision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)
    print("version major={} minor={} vision={} and pic_seen={}".format(major, minor, vision, seen))

    model_dict = model.state_dict()
    key_list = [key for key in model_dict.keys() ]
    num = 6
    length = int(len(key_list)//num)
    pre_index = 0
    for i in range(length+2):
        cur_list = key_list[pre_index:pre_index+num]
        conv_name = cur_list[0]
        conv_layer = model_dict[conv_name]
        filters = conv_layer.shape[0]
        in_dim = conv_layer.shape[1]
        k_size = conv_layer.shape[2]
        conv_shape = (filters,in_dim,k_size,k_size)
        # print("i={} and list={} amd conv_name={} and shape={}".format(i, cur_list,conv_name,conv_shape))
        if len(cur_list) == 6: # with bn
            # darknet bn param:[bias,weight,mean,variance]
            bn_bias = np.fromfile(wf, dtype=np.float32, count= filters)
            model_dict[cur_list[2]].data.copy_( torch.from_numpy(bn_bias))
            bn_weight = np.fromfile(wf, dtype=np.float32, count=filters)
            model_dict[cur_list[1]].data.copy_(torch.from_numpy(bn_weight))
            bn_mean = np.fromfile(wf, dtype=np.float32, count=filters)
            model_dict[cur_list[3]].data.copy_(torch.from_numpy(bn_mean))
            bn_variance = np.fromfile(wf, dtype=np.float32, count=filters)
            model_dict[cur_list[4]].data.copy_(torch.from_numpy(bn_variance))
            # darknet conv param:(out_dim, in_dim, height, width)
            conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
            conv_weights = conv_weights.reshape(conv_shape)
            model_dict[cur_list[0]].data.copy_(torch.from_numpy(conv_weights))
        else:
            conv_bias = np.fromfile(wf, dtype=np.float32, count= filters)
            model_dict[cur_list[1]].data.copy_(torch.from_numpy(conv_bias))
            conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
            conv_weights = conv_weights.reshape(conv_shape)
            model_dict[cur_list[0]].data.copy_(torch.from_numpy(conv_weights))

        pre_index += num
        if i in [57, 65, 73]:
            num = 2
        else:
            num = 6
    assert len(wf.read()) == 0, 'failed to read all data'


if __name__ == "__main__":
    input_size = 416
    num_class = 80
    iou_threshold = 0.45
    score_threshold = 0.3
    #rectangle_colors = (255, 0, 0)

    yolo_ckpt = './model_data/yolov3.weights'
    model = YOLOV3(num_class)
    model.eval()
    read_param_from_file(yolo_ckpt,model)

    image_path = './IMAGES/kite.jpg'
    #image_path = './IMAGES/ocean.jpg'
    original_image = cv2.imread(image_path)
    print(original_image.shape)
    image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
    image_data = image_data.transpose((2, 0, 1))
    image_data = image_data[np.newaxis, ...].astype(np.float32)  #(1,3,416,416)
    input_tensor = torch.from_numpy(image_data).float()

    out_l, out_m, out_s = model(input_tensor)


    # decode
    out_pred = model.predict(out_l,out_m,out_s)
    # post process
    bboxes= postprocess_boxes(out_pred,original_image,input_size=input_size,score_threshold=score_threshold)
    print("before nms box num is ", len(bboxes))
    bboxes = nms(bboxes, iou_threshold, method='nms')
    print("after nms box num is ", len(bboxes))

    # draw
    #image = draw_bbox(original_image, bboxes, CLASSES='./coco/coco_name.txt', rectangle_colors=rectangle_colors)
    image = draw_bbox(original_image, bboxes, CLASSES='./coco/coco_name.txt')
    cv2.imshow("draw",image)
    cv2.imwrite("result/draw.jpg", image)
    cv2.waitKey(0)






