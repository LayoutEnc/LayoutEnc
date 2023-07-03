import argparse
import torch
from get_gradio_demo import get_demo
from utils import load_config, load_transformer, to_numpy


# vg or coco
mode = 'vg'

if mode=='coco':
    object_name_to_idx = {'person': 1, 'bicycle': 2, 'car': 3, 'motorcycle': 4, 'airplane': 5, 'bus': 6, 'train': 7, 'truck': 8, 'boat': 9, 'traffic light': 10, 'fire hydrant': 11, 'stop sign': 13, 'parking meter': 14,
            'bench': 15, 'bird': 16, 'cat': 17, 'dog': 18, 'horse': 19, 'sheep': 20, 'cow': 21, 'elephant': 22, 'bear': 23, 'zebra': 24, 'giraffe': 25, 'backpack': 27, 'umbrella': 28, 'handbag': 31,
            'tie': 32, 'suitcase': 33, 'frisbee': 34, 'skis': 35, 'snowboard': 36, 'sports ball': 37, 'kite': 38, 'baseball bat': 39, 'baseball glove': 40, 'skateboard': 41, 'surfboard': 42,
            'tennis racket': 43, 'bottle': 44, 'wine glass': 46, 'cup': 47, 'fork': 48, 'knife': 49, 'spoon': 50, 'bowl': 51, 'banana': 52, 'apple': 53, 'sandwich': 54, 'orange': 55, 'broccoli': 56,
            'carrot': 57, 'hot dog': 58, 'pizza': 59, 'donut': 60, 'cake': 61, 'chair': 62, 'couch': 63, 'potted plant': 64, 'bed': 65, 'dining table': 67, 'toilet': 70, 'tv': 72, 'laptop': 73,
            'mouse': 74, 'remote': 75, 'keyboard': 76, 'cell phone': 77, 'microwave': 78, 'oven': 79, 'toaster': 80, 'sink': 81, 'refrigerator': 82, 'book': 84, 'clock': 85, 'vase': 86, 'scissors': 87,
            'teddy bear': 88, 'hair drier': 89, 'toothbrush': 90, 'banner': 92, 'blanket': 93, 'branch': 94, 'bridge': 95, 'building-other': 96, 'bush': 97, 'cabinet': 98, 'cage': 99, 'cardboard': 100,
            'carpet': 101, 'ceiling-other': 102, 'ceiling-tile': 103, 'cloth': 104, 'clothes': 105, 'clouds': 106, 'counter': 107, 'cupboard': 108, 'curtain': 109, 'desk-stuff': 110, 'dirt': 111,
            'door-stuff': 112, 'fence': 113, 'floor-marble': 114, 'floor-other': 115, 'floor-stone': 116, 'floor-tile': 117, 'floor-wood': 118, 'flower': 119, 'fog': 120, 'food-other': 121, 'fruit': 122,
            'furniture-other': 123, 'grass': 124, 'gravel': 125, 'ground-other': 126, 'hill': 127, 'house': 128, 'leaves': 129, 'light': 130, 'mat': 131, 'metal': 132, 'mirror-stuff': 133, 'moss': 134,
            'mountain': 135, 'mud': 136, 'napkin': 137, 'net': 138, 'paper': 139, 'pavement': 140, 'pillow': 141, 'plant-other': 142, 'plastic': 143, 'platform': 144, 'playingfield': 145, 'railing': 146,
            'railroad': 147, 'river': 148, 'road': 149, 'rock': 150, 'roof': 151, 'rug': 152, 'salad': 153, 'sand': 154, 'sea': 155, 'shelf': 156, 'sky-other': 157, 'skyscraper': 158, 'snow': 159,
            'solid-other': 160, 'stairs': 161, 'stone': 162, 'straw': 163, 'structural-other': 164, 'table': 165, 'tent': 166, 'textile-other': 167, 'towel': 168, 'tree': 169, 'vegetable': 170,
            'wall-brick': 171, 'wall-concrete': 172, 'wall-other': 173, 'wall-panel': 174, 'wall-stone': 175, 'wall-tile': 176, 'wall-wood': 177, 'water-other': 178, 'waterdrops': 179,
            'window-blind': 180, 'window-other': 181, 'wood': 182, 'other': 183, '__image__': 0, '__null__': 184}

# vg codebook
elif mode=='vg':
    object_name_to_idx = {"__image__": 0, "window": 1, "tree": 2, "man": 3, "shirt": 4, "wall": 5, "person": 6, "building": 7, "ground": 8, "sign": 9,
                        "light": 10, "sky": 11, "head": 12, "leaf": 13, "leg": 14, "hand": 15, "pole": 16, "grass": 17, "hair": 18, "car": 19,
                        "woman": 20, "cloud": 21, "ear": 22, "eye": 23, "line": 24, "table": 25, "shoe": 26, "people": 27, "door": 28, "shadow": 29,
                        "wheel": 30, "letter": 31, "pant": 32, "flower": 33, "water": 34, "chair": 35, "fence": 36, "floor": 37, "handle": 38,
                        "nose": 39, "arm": 40, "plate": 41, "stripe": 42, "rock": 43, "jacket": 44, "hat": 45, "tail": 46, "foot": 47, "face": 48,
                        "road": 49, "tile": 50, "number": 51, "sidewalk": 52, "short": 53, "spot": 54, "bag": 55, "snow": 56, "bush": 57, "boy": 58,
                        "helmet": 59, "street": 60, "field": 61, "bottle": 62, "glass": 63, "tire": 64, "logo": 65, "background": 66, "roof": 67,
                        "post": 68, "branch": 69, "boat": 70, "plant": 71, "umbrella": 72, "brick": 73, "picture": 74, "girl": 75, "button": 76,
                        "mouth": 77, "track": 78, "part": 79, "bird": 80, "food": 81, "box": 82, "banana": 83, "dirt": 84, "cap": 85, "jean": 86,
                        "glasses": 87, "bench": 88, "mirror": 89, "book": 90, "pillow": 91, "top": 92, "wave": 93, "shelf": 94, "clock": 95,
                        "glove": 96, "headlight": 97, "bowl": 98, "trunk": 99, "bus": 100, "neck": 101, "edge": 102, "train": 103, "reflection": 104,
                        "horse": 105, "paper": 106, "writing": 107, "kite": 108, "flag": 109, "seat": 110, "house": 111, "wing": 112, "board": 113,
                        "lamp": 114, "cup": 115, "elephant": 116, "cabinet": 117, "coat": 118, "mountain": 119, "giraffe": 120, "sock": 121,
                        "cow": 122, "counter": 123, "hill": 124, "word": 125, "finger": 126, "dog": 127, "wire": 128, "sheep": 129, "zebra": 130,
                        "ski": 131, "ball": 132, "frame": 133, "back": 134, "bike": 135, "truck": 136, "animal": 137, "design": 138, "ceiling": 139,
                        "sunglass": 140, "sand": 141, "skateboard": 142, "motorcycle": 143, "curtain": 144, "container": 145, "windshield": 146,
                        "cat": 147, "beach": 148, "towel": 149, "knob": 150, "boot": 151, "bed": 152, "sink": 153, "paw": 154, "surfboard": 155,
                        "horn": 156, "pizza": 157, "wood": 158, "bear": 159, "stone": 160, "orange": 161, "engine": 162, "photo": 163, "hole": 164,
                        "child": 165, "railing": 166, "player": 167, "stand": 168, "ocean": 169, "lady": 170, "vehicle": 171, "sticker": 172,
                        "pot": 173, "apple": 174, "basket": 175, "plane": 176, "key": 177, "tie": 178}

@torch.no_grad()
def layout_to_image_generation(model_fn, custom_layout_dict):
    print(custom_layout_dict) # input dataframe
    
    layout_length = 30

    model_kwargs = {
        'bbox': torch.zeros([1, layout_length, 8]),
        'label': torch.zeros([1, layout_length]).long().fill_(object_name_to_idx['__image__']),
    }
    model_kwargs['bbox'][:, :, 2: 4] = 1 # correspond to LayoutEnc dataloader

    for obj_id in range(1, custom_layout_dict['num_obj'] - 1): # row 1-30
        obj_bbox = custom_layout_dict['obj_bbox'][obj_id]
        obj_class = custom_layout_dict['obj_class'][obj_id]
        if obj_class == 'pad':
            obj_class = '__image__'

        x0, y0, x1, y1 = obj_bbox
        
        x_pos0 = round(x0 * (256-1) + 1)
        x_pos1 = round(x1 * (256-1) + 1)
        y_pos0 = round(y0 * (256-1) + 1)
        y_pos1 = round(y1 * (256-1) + 1)

        model_kwargs['bbox'][0][obj_id - 1] = torch.FloatTensor([x0, y0, x1, y1, x_pos0, x_pos1, y_pos0, y_pos1])
        model_kwargs['label'][0][obj_id - 1] = object_name_to_idx[obj_class]

    print(model_kwargs)
    print(model_kwargs['bbox'].shape)

    tmp = 2* torch.randn(1, 3, 256, 256).cuda() - 1

    sample = model_fn(tmp, model_kwargs['label'].cuda(), model_kwargs['bbox'].cuda(), mask=None, mode='val')

    generate_img = to_numpy(sample[0])

    print(generate_img.shape)

    print("sampling complete")

    return generate_img


@torch.no_grad()
def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=str, default='configs/vg.yaml',
                        help='config file path')
    parser.add_argument('--path', type=str, default='data/pretrained/checkpoints/vg/last.ckpt',
                        help='checkpoint path')
    parser.add_argument('--topk_obj', type=int, default=100,
                        help='the topk of object regions')
    parser.add_argument('--topk_bg', type=int, default=100,
                        help='the topk of background regions')
    parser.add_argument('--start', type=int, default=100,
                        help='the topk of background regions')
    parser.add_argument('--end', type=int, default=100,
                        help='the topk of background regions')
    parser.add_argument('--sample_crop_image', type=bool, default=True,
                        help='for computing the sceneFID')
    
    args = parser.parse_args()
    config_path = args.base
    model_path = args.path
    config = load_config(config_path, display=False)
    model = load_transformer(config, ckpt_path=model_path).cuda().eval()    
    
    print("creating model...")
    print(model)

    def model_fn(idx, label, bbox, mask=None, mode='val'):
        return model(idx, label, bbox, mask=mask, mode='val')
    
    print("creating LayoutEnc framework...")

    return model_fn


if __name__ == "__main__":
    model_fn = init()

    demo = get_demo(layout_to_image_generation, model_fn)

    demo.launch(share=False)
