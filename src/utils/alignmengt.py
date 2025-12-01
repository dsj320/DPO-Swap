# copy from https://github.com/rotemtzaban/STIT/blob/main/utils/alignment.py
# Modified to use InsightFace antelopev2 instead of dlib for face detection

import PIL
import PIL.Image
# import dlib
# import face_alignment
import numpy as np
import scipy
import scipy.ndimage
import skimage.io as io
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import cv2
import dlib
import os
from insightface.app import FaceAnalysis
from imutils import face_utils


def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

def get_landmark(filepath, predictor, detector=None, fa=None, insightface_app=None):
    """get landmark with InsightFace antelopev2 for face detection and dlib for landmarks
    :return: np.array shape=(68, 2)
    """
    import dlib
    if fa is not None:
        image = io.imread(filepath)
        lms, _, bboxes = fa.get_landmarks(image, return_bboxes=True)
        if len(lms) == 0:
            return None
        return lms[0]

    # Load image
    if isinstance(filepath, PIL.Image.Image):
        img = np.array(filepath)
        # Convert RGB to BGR for InsightFace
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(filepath)
        if img is None:
            return None
    
    # Use InsightFace for face detection
    if insightface_app is not None:
        try:
            face_infos = insightface_app.get(img)
            if not face_infos:
                return None
            
            # Select the largest face if multiple faces detected
            if len(face_infos) > 1:
                face_info = sorted(face_infos, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
            else:
                face_info = face_infos[0]
            
            # Convert bbox to dlib rectangle format
            bbox = face_info['bbox'].astype(int)
            d = dlib.rectangle(left=bbox[0], top=bbox[1], right=bbox[2], bottom=bbox[3])
            
            # Use dlib for landmark prediction
            shape = predictor(img, d)
            landmark = face_utils.shape_to_np(shape)
            return landmark
            
        except Exception as e:
            print(f"InsightFace detection failed: {e}")
            return None
    
    # Fallback to original dlib detection
    if detector is None:
        detector = dlib.get_frontal_face_detector()
    
    # Convert BGR to RGB for dlib
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img
    
    dets = detector(img_rgb)
    
    for k, d in enumerate(dets):
        shape = predictor(img_rgb, d)
        break
    else:
        return None
    
    t = list(shape.parts())
    a = []
    for tt in t:
        a.append([tt.x, tt.y])
    lm = np.array(a)
    return lm


def align_face(filepath_or_image, predictor, output_size, detector=None,
               enable_padding=False, scale=1.0):
    """
    :param filepath: str
    :return: PIL Image
    """

    c, x, y = compute_transform(filepath_or_image, predictor, detector=detector,
                                scale=scale)
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    img = crop_image(filepath_or_image, output_size, quad, enable_padding=enable_padding)

    # Return aligned image.
    return img


def crop_image(filepath, output_size, quad, enable_padding=False):
    x = (quad[3] - quad[1]) / 2
    qsize = np.hypot(*x) * 2
    # read image
    if isinstance(filepath, PIL.Image.Image):
        img = filepath
    else:
        img = PIL.Image.open(filepath)
    transform_size = output_size
    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink
    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if (crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]):
        img = img.crop(crop)  # (left, upper, right, lower)
        quad -= crop[0:2]
    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
           int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]
    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)
    return img


def compute_transform(filepath, predictor, detector=None, scale=1.0, fa=None, insightface_app=None):
    lm = get_landmark(filepath, predictor, detector, fa, insightface_app)
    if lm is None:
        raise Exception(f'Did not detect any faces in image: {filepath}')
    lm_chin = lm[0: 17]  # left-right
    lm_eyebrow_left = lm[17: 22]  # left-right
    lm_eyebrow_right = lm[22: 27]  # left-right
    lm_nose = lm[27: 31]  # top-down
    lm_nostrils = lm[31: 36]  # top-down
    lm_eye_left = lm[36: 42]  # left-clockwise
    lm_eye_right = lm[42: 48]  # left-clockwise
    lm_mouth_outer = lm[48: 60]  # left-clockwise
    lm_mouth_inner = lm[60: 68]  # left-clockwise
    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg
    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)

    x *= scale
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    return c, x, y


def crop_faces(IMAGE_SIZE, files, scale, center_sigma=0.0, xy_sigma=0.0, use_fa=False, use_insightface=True):
    if use_fa:
        import face_alignment
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True)
        predictor = None
        detector = None
        insightface_app = None
    else:
        fa = None
        predictor = dlib.shape_predictor("Other_dependencies/DLIB_landmark_det/shape_predictor_68_face_landmarks.dat")
        detector = dlib.get_frontal_face_detector()
        
        # Initialize InsightFace if requested
        if use_insightface:
            try:
                model_root = os.environ.get('INSIGHTFACE_MODEL_ROOT', './')
                insightface_app = FaceAnalysis(name='antelopev2', root=model_root, 
                                             providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
                insightface_app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.05)
                print("InsightFace antelopev2 initialized successfully")
            except Exception as e:
                print(f"Failed to initialize InsightFace: {e}")
                print("Falling back to dlib detection")
                insightface_app = None
        else:
            insightface_app = None

    cs, xs, ys = [], [], []
    for  path in tqdm(files):
        c, x, y = compute_transform(path, predictor, detector=detector,
                                    scale=scale, fa=fa, insightface_app=insightface_app)
        cs.append(c)
        xs.append(x)
        ys.append(y)

    cs = np.stack(cs)
    xs = np.stack(xs)
    ys = np.stack(ys)
    if center_sigma != 0:
        cs = gaussian_filter1d(cs, sigma=center_sigma, axis=0)

    if xy_sigma != 0:
        xs = gaussian_filter1d(xs, sigma=xy_sigma, axis=0)
        ys = gaussian_filter1d(ys, sigma=xy_sigma, axis=0)

    quads = np.stack([cs - xs - ys, cs - xs + ys, cs + xs + ys, cs + xs - ys], axis=1)
    quads = list(quads)

    crops, orig_images = crop_faces_by_quads(IMAGE_SIZE, files, quads)

    return crops, orig_images, quads

def crop_faces_from_image(IMAGE_SIZE, frame, scale, center_sigma=0.0, xy_sigma=0.0, use_fa=False, use_insightface=True):
    if use_fa:
        import face_alignment
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True)
        predictor = None
        detector = None
        insightface_app = None
    else:
        import dlib
        fa = None
        predictor = dlib.shape_predictor("Other_dependencies/DLIB_landmark_det/shape_predictor_68_face_landmarks.dat")
        detector = dlib.get_frontal_face_detector()
        
        # Initialize InsightFace if requested
        if use_insightface:
            try:
                model_root = os.environ.get('INSIGHTFACE_MODEL_ROOT', './')
                insightface_app = FaceAnalysis(name='antelopev2', root=model_root, 
                                             providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
                insightface_app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.05)
                print("InsightFace antelopev2 initialized successfully")
            except Exception as e:
                print(f"Failed to initialize InsightFace: {e}")
                print("Falling back to dlib detection")
                insightface_app = None
        else:
            insightface_app = None

    # Process single frame/image (fixed the original bug with undefined 'files')
    c, x, y = compute_transform(frame, predictor, detector=detector,
                                scale=scale, fa=fa, insightface_app=insightface_app)
    cs, xs, ys = [c], [x], [y]

    cs = np.stack(cs)
    xs = np.stack(xs)
    ys = np.stack(ys)
    if center_sigma != 0:
        cs = gaussian_filter1d(cs, sigma=center_sigma, axis=0)

    if xy_sigma != 0:
        xs = gaussian_filter1d(xs, sigma=xy_sigma, axis=0)
        ys = gaussian_filter1d(ys, sigma=xy_sigma, axis=0)

    quads = np.stack([cs - xs - ys, cs - xs + ys, cs + xs + ys, cs + xs - ys], axis=1)
    quads = list(quads)

    # For single image processing, handle directly
    crop = crop_image(frame, IMAGE_SIZE, quads[0].copy())
    if isinstance(frame, str):
        orig_image = Image.open(frame)
    else:
        orig_image = Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame
    crops, orig_images = [crop], [orig_image]

    return crops, orig_images, quads

def crop_faces_by_quads(IMAGE_SIZE, files, quads):
    orig_images = []
    crops = []
    for quad,  path in tqdm(zip(quads, files), total=len(quads)):
        crop = crop_image(path, IMAGE_SIZE, quad.copy())
        orig_image = Image.open(path)
        orig_images.append(orig_image)
        crops.append(crop)
    return crops, orig_images


def calc_alignment_coefficients(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    a = np.matrix(matrix, dtype=float)
    b = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(a.T * a) * a.T, b)
    return np.array(res).reshape(8)