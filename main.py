import sys
import os
import argparse
import numpy as np
import cv2
import time
from PIL import Image
import logging

try:
    from openvino.inference_engine import IECore, IENetwork
except:
    from openvino.ie_api import IECore, IENetwork

fps = ""
framecount = 0
time1 = 0
# help='Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. \
# Sample will look for a suitable plugin for device specified (CPU by default)'
device = 'CPU'

#### color palettes
palette=[]
for i in range(256):
    palette.extend((i,i,i))
palette[:3*21]=np.array([[0, 0, 0],[128, 0, 0],[0, 128, 0],[128, 128, 0],[0, 0, 128],[128, 0, 128],[0, 128, 128],
                        [128, 128, 128],[64, 0, 0],[192, 0, 0],[64, 128, 0],[192, 128, 0],[64, 0, 128],[192, 0, 128],
                        [64, 128, 128],[192, 128, 128],[0, 64, 0],[128, 64, 0],[0, 192, 0],[128, 192, 0],[0, 64, 128]], 
                        dtype='uint8').flatten()
#### yolo class labels
labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']  


#### FOR OPENVINO -----
def load_model(deep_model):
    model_xml = deep_model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    ie = IECore()
    net = ie.read_network(model_xml, model_bin)

    input_info = net.input_info
    input_blob = next(iter(input_info))
    inblob = net.input_info[input_blob]
    inblob.precision = "U8"
    inblob.layout = "NCHW"

    output_blob = next(iter(net.outputs))
    outblob = net.outputs[output_blob]
    outblob.precision = "FP32"
    exec_net = ie.load_network(network=net, device_name=device)
    n, c, h, w = net.input_info[input_blob].input_data.shape
    arr = np.array(net.input_info[input_blob].input_data.shape)

    return n, c, h, w, exec_net, input_blob, output_blob


### FOR YOLOV5 ----
def parse_yolo_region(blob, resized_image_shape, original_im_shape, threshold):
    side=20
    num=3
    anchors = [10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0,
                                198.0,
                                373.0, 326.0]
    try:
        out_blob_n, out_blob_c, out_blob_h, out_blob_w = blob.shape
    except:
        out_blob_n, out_blob_c, out_blob_h, out_blob_w = blob.shape[0]

    predictions = 1.0 / (1.0 + np.exp(-blob))
    assert out_blob_w == out_blob_h, "Invalid size of output blob. It should be in NCHW layout and height should " \
                                     "be equal to width. Current height = {}, current width = {}" \
                                     "".format(out_blob_h, out_blob_w)

    orig_im_h, orig_im_w = original_im_shape
    resized_image_h, resized_image_w = resized_image_shape
    objects = list()

    side_square = side * side
    bbox_size = int(out_blob_c / num)  # 4+1+num_classes

    for row, col, n in np.ndindex(side, side, num):
        bbox = predictions[0, n * bbox_size:(n + 1) * bbox_size, row, col]

        x, y, width, height, object_probability = bbox[:5]
        class_probabilities = bbox[5:]
        if object_probability < threshold:
            continue
        x = (2 * x - 0.5 + col) * (resized_image_w / out_blob_w)
        y = (2 * y - 0.5 + row) * (resized_image_h / out_blob_h)
        
        if int(resized_image_w / out_blob_w) == 8 and int(resized_image_h / out_blob_h) == 8:  # 80x80
            idx = 0
        elif int(resized_image_w / out_blob_w) == 16 and int(resized_image_h / out_blob_h) == 16:  # 40x40
            idx = 1
        elif int(resized_image_w / out_blob_w) == 32 and int(resized_image_h / out_blob_h) == 32:  # 20x20
            idx = 2

        width = (2 * width) ** 2 * anchors[idx * 6 + 2 * n]
        height = (2 * height) ** 2 * anchors[idx * 6 + 2 * n + 1]
        class_id = np.argmax(class_probabilities)
        confidence = object_probability
        objects.append(scale_bbox(x=x, y=y, height=height, width=width, class_id=class_id, confidence=confidence,
                                  im_h=orig_im_h, im_w=orig_im_w, resized_im_h=resized_image_h,
                                  resized_im_w=resized_image_w))
    return objects



def scale_bbox(x, y, height, width, class_id, confidence, im_h, im_w, resized_im_h=640, resized_im_w=640):
    gain = min(resized_im_w / im_w, resized_im_h / im_h)  # gain  = old / new
    pad = (resized_im_w - im_w * gain) / 2, (resized_im_h - im_h * gain) / 2  # wh padding
    x = int((x - pad[0])/gain)
    y = int((y - pad[1])/gain)

    w = int(width/gain)
    h = int(height/gain)
 
    xmin = max(0, int(x - w / 2))
    ymin = max(0, int(y - h / 2))
    xmax = min(im_w, int(xmin + w))
    ymax = min(im_h, int(ymin + h))
    # Method item() used here to convert NumPy types to native types for compatibility with functions, which don't
    # support Numpy types (e.g., cv2.rectangle doesn't support int64 in color parameter)
    return dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=class_id.item(), confidence=confidence.item())


def letterbox(img, size=(640, 640), color=(114, 114, 114)):
    h, w = img.shape[:2]
    new_w, new_h = size

    # Calculate aspect ratios
    aspect_w = new_w / w
    aspect_h = new_h / h
    aspect_ratio = min(aspect_w, aspect_h)

    # Calculate new dimensions
    target_w = int(w * aspect_ratio)
    target_h = int(h * aspect_ratio)
    resized_img = cv2.resize(img, (target_w, target_h))
    canvas = np.full((new_h, new_w, 3), color, dtype=np.uint8)
    x_offset = (new_w - target_w) // 2
    y_offset = (new_h - target_h) // 2

    # Paste the resized image onto the canvas
    canvas[y_offset:y_offset+target_h, x_offset:x_offset+target_w] = resized_img

    return canvas


def intersection_over_union(box_1, box_2):
    width_of_overlap_area = min(box_1['xmax'], box_2['xmax']) - max(box_1['xmin'], box_2['xmin'])
    height_of_overlap_area = min(box_1['ymax'], box_2['ymax']) - max(box_1['ymin'], box_2['ymin'])
    if width_of_overlap_area < 0 or height_of_overlap_area < 0:
        area_of_overlap = 0
    else:
        area_of_overlap = width_of_overlap_area * height_of_overlap_area
    box_1_area = (box_1['ymax'] - box_1['ymin']) * (box_1['xmax'] - box_1['xmin'])
    box_2_area = (box_2['ymax'] - box_2['ymin']) * (box_2['xmax'] - box_2['xmin'])
    area_of_union = box_1_area + box_2_area - area_of_overlap
    if area_of_union == 0:
        return 0
    return area_of_overlap / area_of_union



#### RoadSegmentation
def roadSeg(color_image, n1, c1, h1, w1, exec_net1, input_blob1, output_blob1):
    # Normalization
    prepimg = color_image[:, :, ::-1].copy()
    meta = {'original_shape': color_image.shape, 'resized_shape': (w1, h1)}
    prepimg = cv2.resize(prepimg, (w1, h1))
    prepimg_deep = prepimg.transpose((2, 0, 1)).reshape((1, c1, h1, w1))


    # Predictions
    predictions = exec_net1.infer(inputs={input_blob1: prepimg_deep})
    predictions = predictions[output_blob1].squeeze()
    input_image_height = meta['original_shape'][0]
    input_image_width = meta['original_shape'][1]

    if len(predictions.shape) == 2:  # assume the output is already ArgMax'ed
        result = predictions.astype(np.uint8)
    else:
        result = np.argmax(predictions, axis=0).astype(np.uint8)

    # Masking
    result = cv2.resize(result, (input_image_width, input_image_height), 0, 0, interpolation=cv2.INTER_NEAREST)
    outputimg = Image.fromarray(result, mode="P")
    outputimg.putpalette(palette)
    outputimg = outputimg.convert("RGB")
    outputimg = np.asarray(outputimg)
    outputimg = cv2.cvtColor(outputimg, cv2.COLOR_RGB2BGR)
    imdraw = cv2.addWeighted(color_image, 1.0, outputimg, 0.9, 0)

    return imdraw


#### ObjectDetection
def detectObj(color_image, n2, c2, h2, w2, exec_net2, input_blob2, output_blob2):
    # Normalization
    prepimg = color_image[:, :, ::-1].copy()
    meta = {'original_shape': color_image.shape, 'resized_shape': (w2, h2)}
    prepimg = letterbox(prepimg, size=(w2, h2))
    prepimg_deep = prepimg.transpose((2, 0, 1)).reshape((1, c2, h2, w2))


    # Predictions
    predictions = exec_net2.infer(inputs={input_blob2: prepimg_deep})
    predictions = predictions[output_blob2].squeeze()
    input_image_height = meta['original_shape'][0]
    input_image_width = meta['original_shape'][1]

    # draw detections
    objects = list()
    prob_threshold = 0.4
    iou_threshold = 0.5
    output2  = exec_net2.requests[0].output_blobs 
    for layer_name, out_blob in output2.items():
        objects += parse_yolo_region(out_blob.buffer, prepimg_deep.shape[2:],
                                 color_image.shape[:-1],prob_threshold)

    # Filtering overlapping boxes with respect to the --iou_threshold CLI parameter
    objects = sorted(objects, key=lambda obj : obj['confidence'], reverse=True)
    for i in range(len(objects)):
        if objects[i]['confidence'] == 0:
            continue
        for j in range(i + 1, len(objects)):
            if intersection_over_union(objects[i], objects[j]) > iou_threshold:
                objects[j]['confidence'] = 0

    # Drawing objects with respect to the prob_threshold parameter
    objects = [obj for obj in objects if obj['confidence'] >= prob_threshold]

    origin_im_size = prepimg.shape[:-1]
    for obj in objects:
        color1 = (0,0,255)
        color2 = (255,255,255)
        color3 = (50,50,50)
        class_id = obj['class_id']
        confidence = obj['confidence']
        xmin, ymin, xmax, ymax = obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']        
        color = (0, 255, 0)  # Green color for the bounding box
        label = (' '+str(labels[class_id]) + ' ' + str(round(obj['confidence'] * 100, 1)) + '%')
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2, 1)
        dim, baseline = text_size[0], text_size[1]
        cv2.rectangle(color_image, (obj['xmin'], obj['ymin']), (obj['xmin'] + dim[0] //3, obj['ymin'] - dim[1] + baseline), color3, cv2.FILLED)
        cv2.rectangle(color_image, (xmin, ymin), (xmax, ymax), color1, 2)
        cv2.putText(color_image,label,(xmin, ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color2, 1)


    return color_image


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="input video file.")
    parser.add_argument("--deep_model1", required=True, help="Path of the deeplabv3plus model.xml.")
    parser.add_argument("--deep_model2", required=True, help="Path of the YOLO model.xml.")
    parser.add_argument('--camera_width', type=int, default=640, help='Video frame width. (Default=640)')
    parser.add_argument('--camera_height', type=int, default=640, help='Video frame height. (Default=640)')
    parser.add_argument('--vidfps', type=int, default=30, help='FPS of the output video. (Default=30)')
    args = parser.parse_args()

    video_file = args.video
    camera_width = args.camera_width
    camera_height = args.camera_height
    vidfps = args.vidfps

    cap = cv2.VideoCapture(video_file)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_FPS, vidfps)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec for MP4 format
    out = cv2.VideoWriter('output_video.mp4', fourcc, 10.0, (frame_width, frame_height))  # Output file will be in MP4 format

    waittime = int(1000 / vidfps)  # Delay between frames based on video FPS

    n1, c1, h1, w1, exec_net1, input_blob1, output_blob1 = load_model(args.deep_model1)
    n2, c2, h2, w2, exec_net2, input_blob2, output_blob2 = load_model(args.deep_model2)

    while True:
        t1 = time.perf_counter()

        ret, color_image = cap.read()
        if not ret:
            break

        ### for road seg
        color_image = roadSeg(color_image, n1, c1, h1, w1, exec_net1, input_blob1, output_blob1)
        ### for obj det
        outputimg =  detectObj(color_image, n2, c2, h2, w2, exec_net2, input_blob2, output_blob2)


        # Display
        cv2.putText(outputimg, fps, (camera_width - 170, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow('test', outputimg)
        out.write(outputimg)
        if cv2.waitKey(waittime) & 0xFF == ord('q'):
            break

        # FPS calculation
        framecount += 1
        if framecount >= 10:
            fps = "{:.1f} FPS".format(time1 / 10)
            framecount = 0
            time1 = 0
        t2 = time.perf_counter()
        elapsedTime = t2 - t1
        time1 += 1 / elapsedTime

    cap.release()
    out.release()
    cv2.destroyAllWindows()