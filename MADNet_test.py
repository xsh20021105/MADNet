from __future__ import division
import warnings

from Networks.HR_Net.seg_hrnet import get_seg_model

import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import dataset
import math
from image import *
from utils import *
   
import logging
import nni
from nni.utils import merge_parameter
from config import return_args, args
from scipy.spatial import cKDTree

warnings.filterwarnings('ignore')

setup_seed(args.seed)

logger = logging.getLogger('mnist_AutoML')


def main(args):
    if args['dataset'] == 'NUPT-Metro':
        test_file = '/home/xsh/XSH/MADNet-master/npydata/nupt_test.npy'

    with open(test_file, 'rb') as outfile:
        val_list = np.load(outfile).tolist()

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu_id']
    model = get_seg_model()
    model = nn.DataParallel(model, device_ids=[0])
    model = model.cuda()

    optimizer = torch.optim.Adam(
        [
            {'params': model.parameters(), 'lr': args['lr']},
        ])

    print(args['pre'])

    if not os.path.exists(args['save_path']):
        os.makedirs(args['save_path'])

    if args['pre']:
        if os.path.isfile(args['pre']):
            print("=> loading checkpoint '{}'".format(args['pre']))
            checkpoint = torch.load(args['pre'])
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            args['start_epoch'] = checkpoint['epoch']
            args['best_pred'] = checkpoint['best_prec1']
        else:
            print("=> no checkpoint found at '{}'".format(args['pre']))

    torch.set_num_threads(args['workers'])
    print(args['best_pred'], args['start_epoch'])

    if args['preload_data'] == True:
        test_data = pre_data(val_list, args, train=False)
    else:
        test_data = val_list

    '''inference '''
    prec1, visi = validate(test_data, model, args)

    is_best = prec1 < args['best_pred']
    args['best_pred'] = min(prec1, args['best_pred'])

    print('\nThe visualizations are provided in ', args['save_path'])
    save_checkpoint({
        'arch': args['pre'],
        'state_dict': model.state_dict(),
        'best_prec1': args['best_pred'],
        'optimizer': optimizer.state_dict(),
    }, visi, is_best, args['save_path'])


def pre_data(train_list, args, train):
    print("Pre_load dataset ......")
    data_keys = {}
    count = 0
    for j in range(len(train_list)):
        Img_path = train_list[j]
        fname = os.path.basename(Img_path)
        # print(fname)
        img, fidt_map, kpoint = load_data_fidt(Img_path, args, train)

        blob = {}
        blob['img'] = img
        blob['kpoint'] = np.array(kpoint)
        blob['fidt_map'] = fidt_map
        blob['fname'] = fname
        data_keys[count] = blob
        count += 1

    return data_keys

def validate(Pre_data, model, args):
    print('begin test')
    batch_size = 1
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(Pre_data, args['save_path'],
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),

                            ]),
                            args=args, train=False),
        batch_size=1)

    model.eval()

    mae = 0.0
    mse = 0.0
    accurate = 0.0  
    visi = []
    index = 0

    crowding_count = {
        "严重拥挤": 0,
        "轻度拥挤": 0,
        "基本宽松": 0,
        "舒适": 0
    }

    if not os.path.exists('./local_eval/point_files'):
        os.makedirs('./local_eval/point_files')

    '''output coordinates'''
    f_loc = open("./local_eval/point_files/nupt_localization.txt", "w+")

    for i, (fname, img, fidt_map, kpoint) in enumerate(test_loader):

        count = 0
        img = img.cuda()

        if len(img.shape) == 5:
            img = img.squeeze(0)
        if len(fidt_map.shape) == 5:
            fidt_map = fidt_map.squeeze(0)
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        if len(fidt_map.shape) == 3:
            fidt_map = fidt_map.unsqueeze(0)

        with torch.no_grad():
            d6 = model(img)

            '''return counting and coordinates'''
            # count, pred_kpoint, f_loc = LMDS_counting(d6, i + 1, f_loc, args)
            count, pred_kpoint, f_loc = LMDS_counting_metro(d6, i + 1, f_loc, args)
            point_map = generate_point_map(pred_kpoint, f_loc, rate=1)

            if args['visual'] == True:
                if not os.path.exists(args['save_path'] + '_box/'):
                    os.makedirs(args['save_path'] + '_box/')
                ori_img, box_img = generate_bounding_boxes(pred_kpoint, fname)
                show_fidt = show_map(d6.data.cpu().numpy())
                gt_show = show_map(fidt_map.data.cpu().numpy())
                res = np.hstack((ori_img, gt_show, show_fidt, point_map, box_img))
                cv2.imwrite(args['save_path'] + '_box/' + fname[0], res)

        gt_count = torch.sum(kpoint).item()
        prediction_error = abs(gt_count - count)
        mae += prediction_error
        mse += prediction_error * prediction_error

        if prediction_error <= 1:
            accurate += 1


        show_fidt = show_map(d6.data.cpu().numpy())  
        gray_fidt = cv2.cvtColor(show_fidt, cv2.COLOR_BGR2GRAY)

        keypoint_coords = np.nonzero(pred_kpoint)

        total_keypoint_pixels = 0
        for y, x in zip(keypoint_coords[0], keypoint_coords[1]):
            if 0 <= y < gray_fidt.shape[0] and 0 <= x < gray_fidt.shape[1]:
                total_keypoint_pixels += gray_fidt[y, x]

        total_background_pixels = np.sum(gray_fidt) - total_keypoint_pixels

        
        if (total_keypoint_pixels + total_background_pixels) == 0:
            crowdedness = 0.0
        else:
            crowdedness = total_keypoint_pixels / (total_keypoint_pixels + total_background_pixels)  

        
        crowdedness_percentage = crowdedness * 100 * 1000

       
        if crowdedness >= 0.0003:  # count >= 20
            crowding_status = "严重拥挤"
        elif 0.0002 <= crowdedness < 0.0003:  # 15 <= count < 20
            crowding_status = "轻度拥挤"
        elif 0.000145 <= crowdedness < 0.0002: # 10 <= count < 15
            crowding_status = "基本宽松"
        else:
            crowding_status = "舒适"

        crowding_count[crowding_status] += 1

        print('{fname:<9}  Gt {gt:>2}  Pred {pred:>2}  '
              'KeyPointPixels {keypoints:>8.2f}  BackgroundPixels {background:>10.2f}  '
              'Crowdedness {crowd:>7.4f}% ({status:<4})'.format(
            fname=fname[0], gt=int(gt_count), pred=int(count),
            keypoints=total_keypoint_pixels, background=total_background_pixels,
            crowd=crowdedness_percentage, status=crowding_status))

        visi.append(
            [img.data.cpu().numpy(), d6.data.cpu().numpy(), fidt_map.data.cpu().numpy(),
             fname])
        index += 1

    accuracy = accurate / len(test_loader)
    
    mae = mae * 1.0 / (len(test_loader) * batch_size)
    mse = math.sqrt(mse / (len(test_loader)) * batch_size)

    total_samples = len(test_loader)
    print("\n拥挤程度占比统计:")
    for status, count in crowding_count.items():
        percentage = (count / total_samples) * 100
        print(f"{status}: {percentage:.2f}% ({count}/{total_samples})")

    nni.report_intermediate_result(mae)
    print(' \n* MAE {mae:.3f}\n'.format(mae=mae), '* MSE {mse:.3f}'.format(mse=mse),
          '* Accuracy {accuracy:.3f}'.format(accuracy=accuracy))

    return mae, visi


def LMDS_counting_metro(input, w_fname, f_loc, args):
    """
    输入/输出格式与原函数完全一致
    """
    input_max = torch.max(input).item()

    keep3 = (nn.functional.max_pool2d(input, 3, stride=1, padding=1) == input).float()
    keep5 = (nn.functional.max_pool2d(input, 5, stride=1, padding=2) == input).float()
    keep = keep3 * keep5          
    input = keep * input

    fixed_thr = 115.0 / 255.0 * input_max 

    soft_mask = 0.9 * (input >= fixed_thr).float() + 0.1 * (input >= fixed_thr * 0.8).float()
    input = (input >= fixed_thr).float() * soft_mask


    if input_max < 0.1:
        input.zero_()

    kpoint = input.data.squeeze(0).squeeze(0).cpu().numpy()   
    kpoint = nms_post(kpoint, min_dist=8)                    
    count = int(kpoint.sum())

    f_loc.write('{} {} '.format(w_fname, count))
    return count, kpoint, f_loc


def nms_post(pred_kpoint, min_dist=8):
    ys, xs = np.nonzero(pred_kpoint)
    if ys.size == 0:
        return pred_kpoint
    coords = np.stack([xs, ys], axis=1)
    tree = cKDTree(coords)
    keep = np.ones(len(coords), dtype=bool)
    for i, (x, y) in enumerate(coords):
        if not keep[i]:
            continue
        neighbors = tree.query_ball_point((x, y), r=min_dist)
        conf = pred_kpoint[coords[neighbors][:, 1], coords[neighbors][:, 0]]
        best = neighbors[np.argmax(conf)]
        keep[neighbors] = False
        keep[best] = True
    nms_kp = np.zeros_like(pred_kpoint)
    nxs, nys = coords[keep].T
    nms_kp[nys, nxs] = 1
    return nms_kp



def generate_point_map(kpoint, f_loc, rate=1):
    '''obtain the location coordinates'''
    pred_coor = np.nonzero(kpoint)

    point_map = np.zeros((int(kpoint.shape[0] * rate), int(kpoint.shape[1] * rate), 3), dtype="uint8") + 255  # 22
    # count = len(pred_coor[0])
    coord_list = []
    for i in range(0, len(pred_coor[0])):
        h = int(pred_coor[0][i] * rate)
        w = int(pred_coor[1][i] * rate)
        coord_list.append([w, h])
        cv2.circle(point_map, (w, h), 2, (0, 0, 0), -1)

    for data in coord_list:
        f_loc.write('{} {} '.format(math.floor(data[0]), math.floor(data[1])))
    f_loc.write('\n')

    return point_map

def generate_bounding_boxes(kpoint, fname):
    '''change the data path'''
    Img_data = cv2.imread(
        '/home/xsh/XSH/MADNet-master/NUPT-Metro/test_data/images/' + fname[0])
    ori_Img_data = Img_data.copy()

    '''generate sigma'''
    pts = np.array(list(zip(np.nonzero(kpoint)[1], np.nonzero(kpoint)[0])))
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)

    distances, locations = tree.query(pts, k=4)
    for index, pt in enumerate(pts):
        pt2d = np.zeros(kpoint.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if np.sum(kpoint) > 1:
            sigma = (distances[index][1] + distances[index][2] + distances[index][3]) * 0.1
        else:
            sigma = np.average(np.array(kpoint.shape)) / 2. / 2.  # case: 1 point
        sigma = min(sigma, min(Img_data.shape[0], Img_data.shape[1]) * 0.05)

        if sigma < 6:
            t = 2
        else:
            t = 2
        Img_data = cv2.rectangle(Img_data, (int(pt[0] - sigma), int(pt[1] - sigma)),
                                 (int(pt[0] + sigma), int(pt[1] + sigma)), (0, 255, 0), t)

    return ori_Img_data, Img_data

def show_map(input):
    input[input < 0] = 0
    input = input[0][0]
    fidt_map1 = input
    fidt_map1 = fidt_map1 / np.max(fidt_map1) * 255
    fidt_map1 = fidt_map1.astype(np.uint8)
    fidt_map1 = cv2.applyColorMap(fidt_map1, 2)
    return fidt_map1


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    tuner_params = nni.get_next_parameter()
    logger.debug(tuner_params)
    params = vars(merge_parameter(return_args, tuner_params))
    print(params)

    main(params)
