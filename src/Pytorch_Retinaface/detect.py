import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from math import tan, degrees, atan2
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from statistics import mean
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
import time
import ctypes
from torchsummary import summary
from glob import glob
from tqdm import tqdm
import ctypes
ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')

torch.set_default_tensor_type('torch.cuda.FloatTensor')
parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='./weights/mobilenet0.25_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.04, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=1, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true",
                    default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.65, type=float, help='visualization_threshold')
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    def f(x): return x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

target_aspect_ratio = 1.0
level_eyes = False

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase='test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    # summary(net, (3,640,640))
    print(net)
    
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    device = "cuda"
    net = net.to(device)

    resize = 1
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('id61_0009.mp4')
    # testing begin
    crop = True
    """
    TODO 
    Video Loader
    """
    paths = ('C:/Users/sfrankl/Desktop/504Project/Celeb-df/Celeb-real/**/*.mp4', 'C:/Users/sfrankl/Desktop/504Project/Celeb-df/Celeb-synthesis/**/*.mp4')
    destinations = ('C:/Users/sfrankl/Desktop/504Project/Celeb-df/Celeb-real-cropped/', 'C:/Users/sfrankl/Desktop/504Project/Celeb-df/Celeb-synthesis-cropped/')
    emma = []
    for path, destination in zip(paths,destinations):
        files = glob(path, recursive=True)
        for f in tqdm(files):
            filename = os.path.basename(f)
            frames = []
            bboxes = []
            if os.path.exists(destination + filename):
                continue
            cap = cv2.VideoCapture(f)

            # for wew in range(30):
            while True:
                still_working, img_raw = cap.read()
                if not still_working:
                    break

                img = np.float32(img_raw)

                im_height, im_width, _ = img.shape
                scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
                img -= (104, 117, 123)
                img = img.transpose(2, 0, 1)
                img = torch.from_numpy(img).unsqueeze(0)
                img = img.to(device)
                scale = scale.to(device)

                tic = time.time()
                loc, conf, landms = net(img)  # forward pass
                # print('net forward time: {:.4f}'.format(time.time() - tic))

                priorbox = PriorBox(cfg, image_size=(im_height, im_width))
                priors = priorbox.forward()
                priors = priors.to(device)
                prior_data = priors.data
                boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
                boxes = boxes * scale / resize
                boxes = boxes.cpu().numpy()
                scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
                landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
                scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                    img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                    img.shape[3], img.shape[2]])
                scale1 = scale1.to(device)
                landms = landms * scale1 / resize
                landms = landms.cpu().numpy()

                # ignore low scores
                inds = np.where(scores > args.confidence_threshold)[0]
                boxes = boxes[inds]
                landms = landms[inds]
                scores = scores[inds]

                # keep top-K before NMS
                order = scores.argsort()[::-1][:args.top_k]
                boxes = boxes[order]
                landms = landms[order]
                scores = scores[order]

                # do NMS
                dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
                keep = py_cpu_nms(dets, args.nms_threshold)
                # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
                dets = dets[keep, :]
                landms = landms[keep]

                # keep top-K faster NMS
                dets = dets[:args.keep_top_k, :]
                landms = landms[:args.keep_top_k, :]

                dets = np.concatenate((dets, landms), axis=1)

            # show image
                for b in dets:
                    if b[4] < args.vis_thres:
                        continue
                    if crop:
                        x1, x2 = int(min(b[0], b[2])), int(max(b[0], b[2]))
                        y1, y2 = int(min(b[1], b[3])), int(max(b[1], b[3]))
                        # hr = ((y2-y1)/ (x2-x1)) / target_aspect_ratio
                        # if hr > 1:
                        #     dh = ((hr - 1) * (y2-y1)) / 2
                        #     y1 = int(y1 - dh)
                        #     y2 = int(y2 + dh)
                        # else:
                        #     dw = ((hr - 1) * (x2-x1)) / 2
                        #     x1 = int(x1 - dw)
                        #     x2 = int(x2 + dw)
                        if level_eyes:
                            eye_x1, eye_x2 = b[6], b[8]
                            eye_y1, eye_y2 = b[5], b[7]
                            theta = degrees(atan2(eye_x2-eye_x1, eye_y2-eye_y1))
                            M = cv2.getRotationMatrix2D(((x1 + x2)/2, (y1 + y2) / 2), theta, 1)
                            img_raw = cv2.warpAffine(img_raw, M, img_raw.shape[:2][::-1])
                        frames.append((img_raw))
                        bboxes.append(((y1,y2,x1,x2), y2-y1, x2-x1))
                        continue

                    text = "{:.4f}".format(b[4])
                    b = list(map(int, b))
                    cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (255, 0, 0), 2)
                    cx = b[0]
                    cy = b[1] + 12
                    cv2.putText(img_raw, text, (cx, cy),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                    # landms
                    cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                    cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                    cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                    cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                    cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
                    # save image

                    name = "test.jpg"
                    # cv2.imwrite(name, img_raw)
                    # cv2.imshow(name, img_raw)
                    # k = cv2.waitKey(30) & 0xff
                    # if k == 27:
                    #     break
                    # if k == ord('c'):
                    #     crop = not crop
            heights = [i[1] for i in bboxes]
            widths = [i[2] for i in bboxes]
            # max_height = int(max(bboxes, key = lambda x: x[1]))
            # max_width = int(max(bboxes, key = lambda x: x[2]))
            max_height = max(heights)
            max_width = max(widths)
            ratio = max_height / max_width
            if ratio > target_aspect_ratio:
                max_height = int(target_aspect_ratio * max_width)
                max_width = int(max_width)
            else:
                max_width = int(max_height / target_aspect_ratio)
                max_height = int(max_height)
                
            if max_height % 2 != 0:
                max_height += 1
            if max_width % 2 != 0:
                max_width += 1
            out = cv2.VideoWriter(destination + filename, cv2.VideoWriter_fourcc(*'mp4v'), 28.4, (max_width, max_height))
            for i, (frame, bbox) in enumerate(zip(frames,bboxes)):
                y1,y2,x1,x2 = bbox[0]
                xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
                dh, dw = max_height / 2, max_width / 2
                y1,y2,x1,x2 = int(yc - dh), int(yc + dh), int(xc - dw), int(xc + dw)
                out.write(frame[y1:y2,x1:x2])
                if i == 20:
                    cv2.imwrite('emma_cropped.jpg',frame[y1:y2,x1:x2])
                    cv2.imwrite('emma.jpg',frame)
            out.release()
            cap.release()