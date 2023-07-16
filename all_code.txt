----------------------------------------
-------./backend/project_part.py-------
import numpy as np
import cv2
import os
import open3d as o3d
import json
import numpy as np
from PIL import Image
from pyvirtualdisplay import Display
Display().start()
import argparse


def render_part(
    obj_path,
    part_ids,
    ext_mat,
    int_mat,
    img_width,
    img_height,
    save_path, debug=True):
    
    #Create a file to store all debug information


    # Create a pinhole camera intrinsic
    if debug:
        print("Creating pinhole camera intrinsic...")
    pinhole = o3d.camera.PinholeCameraIntrinsic(
    int(img_width), int(img_height), int_mat[0, 0], int_mat[1, 1], int_mat[0, 2], int_mat[1, 2])

    # Create a perspective camera
    if debug:
        print("Creating perspective camera...")
        
    render = o3d.visualization.rendering.OffscreenRenderer(int(img_width), int(img_height))

    # Load the object
    if debug:
        print("Loading object...")
    render.scene.set_background([255.0, 255.0, 255.0, 255.0])  # RGBA
    
    # Load the object
    if debug:
        print("Loading object...")
    mtl = o3d.visualization.rendering.MaterialRecord()
    mtl.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA
    mtl.shader = "defaultLit"

    # Plot different parts in different colors
    colors = [  [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0],[1.0, 0.0, 1.0, 1.0],[0.0, 1.0, 1.0, 1.0],
    [1.0, 0.0, 0.0, 1.0], [0.5, 0.0, 0.0, 1.0], [0.0, 0.5, 0.0, 1.0], [0.0, 0.0, 0.5, 1.0]]


    idx = 0
    for part in part_ids:

        idx += 1
        curr_obj_path = os.path.join( obj_path,str(part).zfill(2) + '.obj')
        print('curr_obj_path', curr_obj_path)

        mesh = o3d.io.read_triangle_mesh(curr_obj_path)
        mtl.base_color = colors[idx]
        render.scene.add_geometry("Part" + str(part), mesh, mtl)



    render.setup_camera(pinhole, ext_mat)


    try:
        # if no response for 10 seconds, skip

        img_o3d = render.render_to_image()
        print("Image rendered.")
    except Exception as e:
        print(f"Exception occurred during rendering: {e}")



    # If directory does not exist, create it
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    # Overwrite the image if it exists
    if os.path.exists(save_path):
        os.remove(save_path)
    # Save the image
    Image.fromarray(np.array(img_o3d)).save(save_path)


    print("Rendered image saved to " + save_path)
    print("Rendered image: ", img_o3d)

def pose_estimation_and_render_parts(obj_path, ext_mat, int_mat, part_idxs, image_path, output_path):
    img = cv2.imread(image_path)
    width = img.shape[1]
    height = img.shape[0]

    print("Start rendering...")
    render_part(obj_path, part_idxs, ext_mat, int_mat, width, height, output_path)
    print("Done.")

    
   

if __name__ == '__main__':

    # points_2d = []
    # points_3d = []
    # part_idxs = ['0','1','3','2']
    # image_path = 'img_manually.png'
    # output_path = 'output.png'

    # Use argparse to get the arguments
    parser = argparse.ArgumentParser(description='Pose estimation and rendering')
    parser.add_argument('--json', type=str, default='./pose_estimation_data.json', help='path to the json file')

    args = parser.parse_args()
    json_path = args.json

    
    # Read the data from json file
    with open(json_path) as f:
        data = json.load(f)
        ext_mat = np.array(data['extrinsic'])
        int_mat = np.array(data['intrinsic'])
        part_idxs = data['part_idxs']
        image_path = data['image_path']
        output_path = data['output_path']
        obj_path = data['obj_path']
    print("Data loaded from " + json_path)
    print("Extrinsic matrix: ", ext_mat)
    
    # Print image width and height
    img = cv2.imread(image_path)
    width = img.shape[1]
    height = img.shape[0]



    pose_estimation_and_render_parts(obj_path,ext_mat, int_mat, part_idxs, image_path, output_path)

    # Write back the result and success to json file
    data['extrinsic'] = ext_mat.tolist()
    data['intrinsic'] = int_mat.tolist()
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

    print("Done.")

----------------------------------------
-------./backend/pose_estimation.py-------
import numpy as np
import cv2
import os
import open3d as o3d
import json
import numpy as np
from PIL import Image
from pyvirtualdisplay import Display
Display().start()
import argparse



def pose_estimation(k, aux_ext_mat=None):
  '''
  k is a dict containing:
    keypoints_2d:
    keypoints_3d:
    image_width:
    image_height:
  '''

  keypoints_2d = np.array(k['keypoints_2d'], dtype=np.float32)
  keypoints_3d = np.array(k['keypoints_3d'], dtype=np.double)
  if not (keypoints_2d.shape[1] == 2 and keypoints_3d.shape[1] == 3 and keypoints_2d.shape[0] == keypoints_3d.shape[0]):
    return False, f"Keypoints shape mismatch. {keypoints_2d.shape=} {keypoints_3d.shape=}", {}

  n = keypoints_2d.shape[0]
  if n <= 3:
    return False, "Not enough keypoints.", {}

  cx = k['image_width'] / 2
  cy = k['image_height'] / 2

  best_reproj_err = float('inf')
  result = {}
  for f in range(300, 2000, 10):
    camera_matrix = np.array([
       [f, 0, cx],
       [0, f, cy],
       [0, 0, 1]
     ], dtype=np.float32)
    dist_coeffs = np.zeros((4,1))
    # success, rotation_vector, translation_vector = cv2.solvePnP(keypoints_3d, keypoints_2d, camera_matrix, dist_coeffs, flags=0)
    # success, rotation_vector, translation_vector, _ = cv2.solvePnPRansac(keypoints_3d, keypoints_2d, camera_matrix, dist_coeffs, flags=0)
    if aux_ext_mat is not None:
      rotation_vector_init = cv2.Rodrigues(aux_ext_mat[:3, :3])[0]
      translation_vector_init = np.array(aux_ext_mat[:3, [3]])
      success, rotation_vector, translation_vector, _ = cv2.solvePnPRansac(keypoints_3d, keypoints_2d, camera_matrix, dist_coeffs, 
      # success, rotation_vector, translation_vector = cv2.solvePnP(keypoints_3d, keypoints_2d, camera_matrix, dist_coeffs, 
      rotation_vector_init, translation_vector_init,
      useExtrinsicGuess=True,
      flags=0)
    else:
      success, rotation_vector, translation_vector, _ = cv2.solvePnPRansac(keypoints_3d, keypoints_2d, camera_matrix, dist_coeffs, flags=0)
      # success, rotation_vector, translation_vector = cv2.solvePnP(keypoints_3d, keypoints_2d, camera_matrix, dist_coeffs, flags=0)
    rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
    ext_matrix = np.zeros((4, 4))
    ext_matrix[:3, :3] = rotation_matrix
    ext_matrix[:3, 3] = translation_vector[:, 0]
    ext_matrix[3, 3] = 1

    reprojected = cv2.projectPoints(keypoints_3d, rotation_vector, translation_vector, camera_matrix, dist_coeffs)[0]
    reprojected = reprojected[:, 0, :]
    reproj_err = ((keypoints_2d - reprojected) ** 2).sum(axis=-1).mean()
    
    if success and reproj_err < best_reproj_err:
      best_reproj_err = reproj_err
      result['reprojected'] = reprojected.tolist()
      result['extrinsic'] = ext_matrix.tolist()
      result['reprojection_error'] = reproj_err
      result['intrinsic'] = camera_matrix.tolist()
    
  return success, "Success." if len(result) != 0 else "SolvePnP failed.", result

def render_part(
    obj_path,
    part_ids,
    ext_mat,
    int_mat,
    img_width,
    img_height,
    save_path, debug=True):
    
    #Create a file to store all debug information


    # Create a pinhole camera intrinsic
    if debug:
        print("Creating pinhole camera intrinsic...")
    pinhole = o3d.camera.PinholeCameraIntrinsic(
    int(img_width), int(img_height), int_mat[0, 0], int_mat[1, 1], int_mat[0, 2], int_mat[1, 2])

    # Create a perspective camera
    if debug:
        print("Creating perspective camera...")
        
    render = o3d.visualization.rendering.OffscreenRenderer(int(img_width), int(img_height))

    # Load the object
    if debug:
        print("Loading object...")
    render.scene.set_background([255.0, 255.0, 255.0, 255.0])  # RGBA
    
    # Load the object
    if debug:
        print("Loading object...")
    mtl = o3d.visualization.rendering.MaterialRecord()
    mtl.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA
    mtl.shader = "defaultLit"

    # Plot different parts in different colors
    colors = [  [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0],[1.0, 0.0, 1.0, 1.0],[0.0, 1.0, 1.0, 1.0],
    [1.0, 0.0, 0.0, 1.0], [0.5, 0.0, 0.0, 1.0], [0.0, 0.5, 0.0, 1.0], [0.0, 0.0, 0.5, 1.0]]


    idx = 0
    for part in part_ids:

        idx += 1
        curr_obj_path = os.path.join( obj_path,str(part).zfill(2) + '.obj')
        print('curr_obj_path', curr_obj_path)

        mesh = o3d.io.read_triangle_mesh(curr_obj_path)
        mtl.base_color = colors[idx]
        render.scene.add_geometry("Part" + str(part), mesh, mtl)



    render.setup_camera(pinhole, ext_mat)


    try:
        # if no response for 10 seconds, skip

        img_o3d = render.render_to_image()
        print("Image rendered.")
    except Exception as e:
        print(f"Exception occurred during rendering: {e}")



    # If directory does not exist, create it
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    # Overwrite the image if it exists
    if os.path.exists(save_path):
        os.remove(save_path)
    # Save the image
    Image.fromarray(np.array(img_o3d)).save(save_path)


    print("Rendered image saved to " + save_path)
    print("Rendered image: ", img_o3d)

def pose_estimation_and_render_parts(obj_path, points_2d, points_3d, part_idxs, image_path, output_path):
    img = cv2.imread(image_path)
    width = img.shape[1]
    height = img.shape[0]



    k = {
    'keypoints_2d': points_2d,
    'keypoints_3d': points_3d,
    'image_width': width,
    'image_height': height
    }
    success, msg, result = pose_estimation(k)

    print(msg)
    print(result)

    ext_mat = np.array(result['extrinsic'])
    int_mat = np.array(result['intrinsic'])
    print("Start rendering...")
    render_part(obj_path, part_idxs, ext_mat, int_mat, width, height, output_path)
    print("Done.")

    return success, ext_mat, int_mat
   

if __name__ == '__main__':

    # points_2d = []
    # points_3d = []
    # part_idxs = ['0','1','3','2']
    # image_path = 'img_manually.png'
    # output_path = 'output.png'

    # Use argparse to get the arguments
    parser = argparse.ArgumentParser(description='Pose estimation and rendering')
    parser.add_argument('--json', type=str, default='./pose_estimation_data.json', help='path to the json file')

    args = parser.parse_args()
    json_path = args.json
    
    # Read the data from json file
    with open(json_path) as f:
        data = json.load(f)
        points_2d = data['points_2d']
        points_3d = data['points_3d']
        part_idxs = data['part_idxs']
        image_path = data['image_path']
        output_path = data['output_path']
        obj_path = data['obj_path']
    
    # Print image width and height
    img = cv2.imread(image_path)
    width = img.shape[1]
    height = img.shape[0]

    print(points_2d)

    for i in range(len(points_2d)):
        points_2d[i][0] = points_2d[i][0] * width
        points_2d[i][1] = points_2d[i][1] * height
    
    # # Convert to only 3 digits after the decimal point
    points_2d = [[round(x, 3), round(y, 3)] for (x, y) in points_2d]
    points_3d = [[round(x, 3), round(y, 3), round(z, 3)] for (x, y, z) in points_3d]
    print(points_2d)
    print(points_3d)


    success, ext_mat, int_mat = pose_estimation_and_render_parts(obj_path,points_2d, points_3d, part_idxs, image_path, output_path)

    # Write back the result and success to json file
    data['success'] = success
    data['extrinsic'] = ext_mat.tolist()
    data['intrinsic'] = int_mat.tolist()
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

    print("Done.")

----------------------------------------
-------./backend/app.py-------
from flask import Flask, request, jsonify , send_file
from flask_cors import CORS
# from pose_estimation import pose_estimation_and_render_parts
import json
import subprocess
import math
from segment_anything import sam_model_registry
import os
import locale
import torch
import cv2
import numpy as np
from pycocotools import mask as mask_utils
# Use matplotlib.use('Agg') to avoid the error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.data_utils import get_data, get_data, get_video, write_data, is_same_parts
#  Import ExtendedSamPredictor from ./SAM/sam.py
from SAM.sam import ExtendedSamPredictor, show_mask, show_points, get_inner_points, remove_mask_overlap

# Set the environment variable to explicitly set the locale to UTF-8.
# To avoid: “Fatal Python error: config_get_locale_encoding: failed to get the locale encoding: nl_langinfo(CODESET) failed”。
os.environ["LC_ALL"] = "en_US.UTF-8"
os.environ["LANG"] = "en_US.UTF-8"
os.environ["LANGUAGE"] = "en_US.UTF-8"
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
port = 5000

# Rest of your code
DATA_JSON_PATH = './new_data.json'

# Load model here to avoid loading model every time
CHECKPOINT_PATH = os.path.join(os.getcwd() + '/SAM', "weights", "sam_vit_h_4b8939.pth")
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)

# Json address for pose estimation
POSE_ESTIMATION_DATA_PATH = './pose_estimation_data.json'
SEGMENTATION_DATA_PATH = './segmentation_data.json'

app = Flask(__name__)
CORS(app) # This will enable CORS for all routes



#################### Image Rendering ####################
@app.route('/image/<path:filename>')
def get_image(filename):
    image_path = os.path.join(filename)

    if os.path.exists(image_path):
        # Add a timestamp to the image URL
        response = send_file(image_path, mimetype='image/jpg')
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response
    else:
        return 'File not found', 404



#################### Pose Estimation ####################
@app.route('/pose-estimation', methods=['POST'])
def pose_estimation():
    datas = request.json
    # make coord 2d and 3d from dict to list:
    points_2d = []
    points_3d = []
    for data in datas["3d-coordinates"]:
        # TODO: if len == 0 
        point_3d = (data['x'], data['y'],data['z'])
        points_3d.append(point_3d)

    for data in datas["2d-coordinates"]:
        point_2d = (data['x'], data['y'])
        points_2d.append(point_2d)
    
    catergory = datas["Category"]
    name = datas["SubCategory"]
    step_id = datas["Object"]
    current_img = datas["image-path"]


    currentModelFilePaths = datas["currentModelFilePaths"]

    #e.g. parts = "0,1,2"
    parts = [currentModelFilePath.split('/')[-1].split('.')[0] for currentModelFilePath in currentModelFilePaths]
    if len(parts) == 0:
        print(currentModelFilePaths)
        return jsonify({"error": "No part selected."})
    # change 00 -> 0, 01 -> 1, 02 -> 2
    parts = [str(int(part)) for part in parts]
    

    # frame_num = datas["image-path"].split('/')[-1].split('_')[-1].split('.')[0]
    
    image_path = os.path.join(os.getcwd(), '..', 'public' ,current_img)
    output_path  = os.path.join(os.getcwd(), 'dynamic_dataset/pose-estimation' , catergory, name, step_id)
    obj_path = os.path.join(os.getcwd(), '..', 'public/dataset/parts' , catergory, name)


    print("image_path: ", image_path)
    print("output_path: ", output_path)
    ##### Check if the parts overlap with previous frame #####
    data = {}
    data['previous_frame_data'] = {}
    data['same_parts_exist'] = False
    json_data = get_data(DATA_JSON_PATH)
    for prev_data in json_data:
        if prev_data["category"] == catergory and prev_data["name"] == name:
            step_id = current_img.split('/')[-2].split('_')[-1]
            video_id = current_img.split('/')[-1].split('_')[0]
            frame_id = current_img.split('/')[-1].split('_')[-1].split('.')[0]

            for video in prev_data['steps'][int(step_id)]['video']:
                if video['video_id'].split('watch?v=')[-1] == video_id:
                    for frame in video['frames']:
                        print("frame: ", frame)
                        if int(frame['frame_id']) == int(frame_id):
                            data['previous_frame_data']=frame
                            for i in range(len(frame['parts'])):
                                if is_same_parts(','.join(parts), frame['parts'][i]) and frame['extrinsics'][i] != []:
                                    data['same_parts_exist'] = True
                                    
                            break
                    break
            break
    part_ls = []
    print(data['previous_frame_data'])
    if len(data['previous_frame_data']) > 0:
        print("previous_frame_data: ", data['previous_frame_data'])
        for part in data['previous_frame_data']['parts']:
            part_ls.append(part.split(','))

        # if parts has overlap with previous frame, raise error
        for part in parts:
            if part in part_ls:
                return jsonify({"success": False, "msg": "Part overlap with previous frame." ,"imagePath": image_path})
    
    # if parts has no overlap with previous frame, and part_ls is not empty, we need to show the previous frame
    data['show_previous_frame'] = True if len(part_ls) > 0 else False
            

    # save points_2d, points_3d, part_idxs, image_path, output_path to json file
    
    data["points_2d"] = points_2d
    data["points_3d"] = points_3d
    data["part_idxs"] = ','.join(parts)
    data["image_path"] = image_path
    if data['same_parts_exist']:
        data["output_path"] = os.path.join(output_path, datas["image-path"].split('/')[-1].split('.')[0]+'_parts'+ ''.join(parts) + '_new.jpg')
    else:
        data["output_path"] = os.path.join(output_path, datas["image-path"].split('/')[-1].split('.')[0]+'_parts'+ ''.join(parts) + '.jpg')
    data["obj_path"] = obj_path
    data["category"] = catergory
    data["name"] = name
    data["step_id"] = step_id
    data["frame_id"] = datas["image-path"].split('/')[-1].split('_')[-1].split('.')[0]
    data["video_id"] = datas["image-path"].split('/')[-1].split('_')[0]
    with open(POSE_ESTIMATION_DATA_PATH, 'w') as outfile:
        json.dump(data, outfile, indent=4)

    print("Start pose estimation...")
    cmd_to_execute = 'python ' + './pose_estimation.py' + ' --json ' + POSE_ESTIMATION_DATA_PATH
    # Construct the Conda command
    conda_cmd = f'conda run -n IKEA-dataset {cmd_to_execute}'

    ret = subprocess.run(conda_cmd, shell=True, capture_output=True, text=True)

    # ret = subprocess.run(cmd_to_execute, shell=True, capture_output=True, text=True)
    print(" pose estimation finished")

    return_code = ret.returncode
    if "SolvePnP failed" in ret.stdout:
        print ("SolvePnP failed!!")
        return jsonify({"error": "SolvePnP failed."}), 500 #return the original image path
    elif "Not enough keypoints" in ret.stdout:
        print ("Not enough keypoints!!")
        return jsonify({"error": "Not enough keypoints."}), 500
    elif "Keypoints shape mismatch" in ret.stdout:
        print ("Keypoints shape mismatch!!")
        return jsonify({"error": "Keypoints shape mismatch."}), 500


    print(ret)
    print(return_code)

    # Return output image path after ./public
    output_path = 'http://localhost:' + str(port) +'/image/' + output_path
    print("output_path: ", output_path)

    # Read ext_mat and int_mat from pose_estimation json file
    # with open(pose_estimation_json_path) as f:
    with open(POSE_ESTIMATION_DATA_PATH) as f:
        pose_estimation_data = json.load(f)
    ext_mat = pose_estimation_data["extrinsic"]
    int_mat = pose_estimation_data["intrinsic"]

    print("ext_mat: ", ext_mat)
    print("int_mat: ", int_mat)
    return jsonify({"imagePath": output_path, "messages": ["Pose estimation finished!"]}), 200


#################### Segmentation ####################
def segmentation_sam():
    with open(SEGMENTATION_DATA_PATH, 'r') as f:
        data = json.load(f)

    image_path = data['image_path']
    ouptut_path = data['output_path']
    input_point = np.array(eval(data['positive_points']) + eval(data['negative_points']))
    input_label = np.array([1] * len(eval(data['positive_points'])) + [0] * len(eval(data['negative_points'])))
    image_embedding_path = data['image_embedding_path']
    print(input_point)
    print(input_label)


    image_bgr = cv2.imread(image_path)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    original_image = image

    sam.to(device=DEVICE)
    image_embedding = np.load(image_embedding_path)

    # Create a new predictor
    new_predictor = ExtendedSamPredictor(sam)

    # Get the original size
    original_size = original_image.shape[:2]  # Height and width of the original image

    # Apply the transform to get the input size
    input_image = new_predictor.transform.apply_image(original_image)
    input_size = input_image.shape[:2]  # Input size after transformation

    # Load the saved embedding and sizes
    new_predictor.load_image_embedding(image_embedding, original_size, input_size)


    # The points coordinates are in the range [0, 1], and should be rescaled to the image size
    h, w = image.shape[:2]
    input_point[:, 0] *= w
    input_point[:, 1] *= h

    if data['previous_mask_exist']:
        previous_masks = data['previous_masks']
        # Convert the previous mask to negative points
        negative_points_from_previous_mask = []
        for prev_mask in previous_masks:
            if prev_mask == {}:
                continue
            else:
                rle = {
                    'counts': prev_mask['counts'].encode('ascii'),
                    'size': prev_mask['size'],
                }

                mask = mask_utils.decode(prev_mask)
                inner_points = get_inner_points(mask)
                negative_points_from_previous_mask.append(inner_points)
        negative_points_from_previous_mask = np.concatenate(negative_points_from_previous_mask, axis=0)

        # Add the negative points from the previous mask to the input points
        input_point = np.concatenate([input_point, negative_points_from_previous_mask], axis=0)
        input_label = np.concatenate([input_label, np.zeros(len(negative_points_from_previous_mask))], axis=0)
                            


    masks, scores, logits = new_predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
        return_logits = True
    )
    
    mask_threshold = 0.5
    
    logits = logits[np.newaxis, :, :]
    resized_mask_torch = sam.postprocess_masks(torch.from_numpy(logits), input_size, original_size)
    binary_mask = (resized_mask_torch[0][0].cpu().numpy() > mask_threshold).astype(np.uint8)

    # Remove overlapping with previous masks
    if data['previous_mask_exist']:
        binary_mask = remove_mask_overlap(previous_masks, binary_mask)
    
    ########## Image with original frame image ##########
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    # show_mask(masks[0], plt.gca())
    show_mask(binary_mask, plt.gca())
    # show_points(input_point, input_label, plt.gca())
    plt.gca().set_position([0, 0, 1, 1])
    plt.axis('off') # Remove the axis   

    # Save the figure
    # Make directory if it doesn't exist
    output_dir = os.path.join(os.getcwd(), '/'.join(ouptut_path.split("/")[:-1]))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(os.getcwd(), ouptut_path.split('.')[0]+'_with_frame_img.jpg'), bbox_inches='tight', pad_inches=0)
    plt.close()

    ########## Image with points ##########
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_points(input_point, input_label, plt.gca())
    show_mask(binary_mask, plt.gca())
    plt.gca().set_position([0, 0, 1, 1])
    plt.axis('off') # Remove the axis

    # Save the figure
    plt.savefig(os.path.join(os.getcwd(), ouptut_path.split('.')[0]+'_with_points.jpg'), bbox_inches='tight', pad_inches=0)
    plt.close()

    ########## Mask only ##########

    plt.figure(figsize=(10,10))
    plt.imshow(binary_mask)
    plt.gca().set_position([0, 0, 1, 1])
    plt.axis('off') # Remove the axis

    # Save the figure
    # Make directory if it doesn't exist
    plt.savefig(os.path.join(os.getcwd(), ouptut_path), bbox_inches='tight', pad_inches=0)

    # Save the mask to a json file using RLE encoding to the input json
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))

    # Convert counts to ASCII string
    rle['counts'] = rle['counts'].decode('ascii')
    data['new_mask']={"size": rle['size'], "counts": rle['counts']}
    with open(SEGMENTATION_DATA_PATH, 'w') as f:
        json.dump(data, f, indent=4)
    
    return 'success'
   

@app.route('/segmentation', methods=['POST'])
def segmentation():
    datas = request.json
    # JSON.stringify({"positive-keypoints": positiveKeypoints, "negative-keypoints": negativeKeypoints}) // sending the coordinates

    # convert coordinates from dict to list
    positiveKeypoints = []
    negativeKeypoints = []
    for data in datas["positive-keypoints"]:
        point = [data['x'], data['y']]
        positiveKeypoints.append(point)
    for data in datas["negative-keypoints"]:
        point = [data['x'], data['y']]
        negativeKeypoints.append(point)
    
    if positiveKeypoints == [] and negativeKeypoints == []:
        return jsonify({'error': 'No points provided'}), 400
    
    image_path = datas["image-path"]
    category = datas["Category"]
    name = datas["SubCategory"]
    catergory = datas["Category"]
    name = datas["SubCategory"]
    current_img = datas["image-path"]


    # Change the list to string "[[123,234], [234,345]]" , str() not working, use json.dumps()
    positiveKeypoints = str(positiveKeypoints)
    negativeKeypoints = str(negativeKeypoints)

    print( "Creating segmentation data for image: ", image_path)
    
    # Create a json for segmentation
    output_file = {}
    output_file['image_path'] = os.path.join(os.getcwd(), '..', 'public' ,image_path)
    output_file['image_embedding_path'] = os.path.join(os.getcwd(),'dynamic_dataset/image_embeddings' , category, name, datas['Object'], image_path.split('/')[-1].split('.')[0] + '.npy')
    output_file['positive_points'] = positiveKeypoints
    output_file['negative_points'] = negativeKeypoints

    
    currentModelFilePaths = datas["currentModelFilePaths"]
    #e.g. parts = "0,1,2"
    parts = [currentModelFilePath.split('/')[-1].split('.')[0] for currentModelFilePath in currentModelFilePaths]
    if len(parts) == 0:
        print(currentModelFilePaths)
        return jsonify({"error": "No part selected."})
    # change 00 -> 0, 01 -> 1, 02 -> 2
    parts = [str(int(part)) for part in parts]

    # Check if previous mask exists
    count = 0
    data = {}
    data['previous_frame_data'] = [{}]
    data['same_parts_exist'] = False
    json_data = get_data(DATA_JSON_PATH)
    for prev_data in json_data:
        if prev_data["category"] == catergory and prev_data["name"] == name:
            step_id = current_img.split('/')[-2].split('_')[-1]
            video_id = current_img.split('/')[-1].split('_')[0]
            frame_id = current_img.split('/')[-1].split('_')[-1].split('.')[0]

            for video in prev_data['steps'][int(step_id)]['video']:
                if video['video_id'].split('watch?v=')[-1] == video_id:
                    for frame in video['frames']:
                        if int(frame['frame_id']) == int(frame_id):
                            data['previous_frame_data'] = frame
                           
                            # count the number of masks that not == {}
                            for mask in frame['mask']:
                                if mask != {}:
                                    count += 1
                            for i in range(len(frame['parts'])):
                                if ','.join(parts) == frame['parts'][i]:
                                    # remove this mask
                                    if frame['mask'][i] != {}:
                                        count -= 1
                                        data['same_parts_exist'] = True

                                    data['previous_frame_data']['mask'][i] = {}
                                    
                                    print(count)
                                    print("Previous mask parts overlap with current mask, no previous mask loaded for parts: ", parts)
                                    
                            break
                    break
            break
    part_ls = []
    print(data['previous_frame_data'])
    if len(data['previous_frame_data']) > 0:
        print("previous_frame_data: ", data['previous_frame_data'])
        for part in data['previous_frame_data']['parts']:
            part_ls.append(part.split(','))
            for part in parts:
                if part in part_ls:
                    return jsonify({"success": False, "msg": "Previous mask parts overlap with current mask." ,"imagePath": image_path})

    if count > 0:
        output_file['previous_mask_exist'] = True
        output_file['previous_masks'] = data['previous_frame_data']['mask']
        output_file['previous_parts'] = data['previous_frame_data']['parts']
    else:
        output_file['previous_mask_exist'] = False

    output_file['parts'] = ','.join(parts)
    output_file["category"] = catergory
    output_file["name"] = name
    output_file["step_id"] = step_id
    output_file["frame_id"] = datas["image-path"].split('/')[-1].split('_')[-1].split('.')[0]
    output_file["video_id"] = datas["image-path"].split('/')[-1].split('_')[0]
    if data['same_parts_exist']:
        output_file['output_path']  =  os.path.join(os.getcwd(),'dynamic_dataset/masks' , category, name, datas["Object"], image_path.split('/')[-1].split('.')[0]+'_parts'+ ''.join(parts) +'_new.jpg')
    else:
        output_file['output_path']  =  os.path.join(os.getcwd(),'dynamic_dataset/masks' , category, name, datas["Object"], image_path.split('/')[-1].split('.')[0]+'_parts'+ ''.join(parts) +'.jpg')

    
    with open(SEGMENTATION_DATA_PATH, 'w') as outfile:
        json.dump(output_file, outfile , indent=4)

    # Run segmentation
    print("Start segmentation...")
    success = segmentation_sam()
    print("Segmentation finished.")

    # Return output image path after ./public
    # Get absolute path of output image
    output_path = 'http://localhost:' + str(port) + '/image/' +os.path.abspath(output_file['output_path'])
    print("output_path: ", output_path)

    return jsonify({"imagePath": output_path, "messages": ['Segmentation finished!']})


#################### Table of Contents ####################
# get-categories, get-subcategories, get-steps, get-object
@app.route('/get-categories', methods=['GET'])
def get_categories():
    # Categories can be extracted from the json data
    categories = []
    json_data = get_data(DATA_JSON_PATH)
    for data in json_data:
        if data['category'] not in categories:
            categories.append(data['category'])

    # print("categories: ", categories)
    return jsonify({"categories": categories})
    

@app.route('/get-segmentation-image-path-initial', methods=['POST'])
def get_segmentationImagePath_initial():
    datas = request.json
    selected_category = datas["Category"]
    selected_subcategory = datas["SubCategory"]
    selected_step = datas["Object"]
    selected_frame = datas["Frame"].split('/')[-1]
    original_image_path = datas["OriginalImagePath"]

    frame_num = int(selected_frame.split('_')[-1].split('.')[0])

    # Check if the segmentation and pose estimation image exists
    # If exists, return the path
    # If not, return the original image path

    segmentation_image_dir = os.path.join('dynamic_dataset/masks' , selected_category, selected_subcategory, selected_step)
    projection_image_dir = os.path.join( 'dynamic_dataset/pose-estimation' , selected_category, selected_subcategory, selected_step)
    return_info = {}
    print ("segmentation_image_dir: ", segmentation_image_dir)
    print ("projection_image_dir: ", projection_image_dir)
    # if the dir is not empty, return the list of segmentation images
    segmentationImagePaths = []
    if os.path.exists(segmentation_image_dir):
        if len(os.listdir(segmentation_image_dir)) > 0:
            print(os.listdir(segmentation_image_dir))
            for file in os.listdir(segmentation_image_dir):
                print('os.listdir(segmentation_image_dir) ', os.listdir(segmentation_image_dir))
                print('frame_num ', frame_num)
                print('file.split(_)[2]', file.split('_')[2])
                if file.split('_')[2]== str(frame_num):
                    print ("segmentation image exists")

                    segmentationImagePaths.append('http://localhost:' + str(port) + '/image/' +os.path.join(segmentation_image_dir, file))
            return_info["segmentationImagePaths"] = segmentationImagePaths
    if "segmentationImagePaths" not in return_info:
        return_info["segmentationImagePaths"] = [original_image_path]


    # if the dir is not empty, return the list of projection images
    projectImagePaths = []
    if os.path.exists(projection_image_dir):
        if len(os.listdir(projection_image_dir)) > 0:
            print ("projection image exists")
           
            for file in os.listdir(projection_image_dir):
                if file.split('_')[-2]== str(frame_num):
                    print ("segmentation image not exists")

                    projectImagePaths.append('http://localhost:' + str(port) + '/image/' +os.path.join(projection_image_dir, file))
            return_info["projectImagePaths"] = projectImagePaths
    if "projectImagePaths" not in return_info:
        print ("projection image not exists")
        return_info["projectImagePaths"] = [original_image_path]
    
    print ("return_info: ", return_info)
    return jsonify(return_info)


@app.route('/get-subcategories', methods=['POST'])
def get_subcategories():
    # Subcategories can be extracted from the json data
    datas = request.json
    selected_category = datas["category"]
    subcategories = []
    json_data = get_data(DATA_JSON_PATH)
    for data in json_data:
        if data['category'] == selected_category:
            if data['name'] not in subcategories:
                subcategories.append(data['name'])
    # print("subcategories: ", subcategories)
    return jsonify({"subcategories": subcategories})


@app.route('/get-steps', methods=['POST'])
def get_steps():
    # Objects can be extracted from the json data
    datas = request.json
    selected_category = datas["category"]
    selected_subcategory = datas["subCategory"]
    objects = []
    filePaths = []
    
    parts_base_path = os.path.join('../public/dataset/parts', selected_category, selected_subcategory)
    parts_base_to_return = os.path.join('./dataset/parts', selected_category, selected_subcategory)
    
    json_data = get_data(DATA_JSON_PATH)
    for data in json_data:
        if data['category'] == selected_category and data['name'] == selected_subcategory:
            for step in data['steps']:
                if step['step_id'] not in objects:
                    objects.append('step_'+ str(step['step_id']))
    # Find all avaliable files that ends with .obj in the parts_base_path using os
    obj_num_ls = []
    for file in os.listdir(parts_base_path):
        if file.endswith(".obj"):
            filePaths.append(os.path.join(parts_base_to_return, file))
            obj_num_ls.append(int(file.split('_')[-1].split('.')[0]))
    # Sort the file paths based on the obj number
    filePaths = [x for _,x in sorted(zip(obj_num_ls, filePaths))]
    
    # print("objects: ", objects)

    return jsonify({"objects": objects, "filePaths": filePaths})

@app.route('/get-object', methods=['POST'])
def get_object():
    # Objects can be extracted from the json data
    # TODO: get object from json data
    datas = request.json
    selected_category = datas["category"]
    selected_subcategory = datas["subCategory"]
    selected_step = datas["object"]

    originalImagePaths = []
    image_base_path = os.path.join('./dataset/frames', selected_category, selected_subcategory, selected_step)
    # Go through the directory and find all the images
    frame_number_ls = []
    for filename in os.listdir(os.path.join("../public/",image_base_path)):
        if filename.endswith(".jpg"):
            originalImagePaths.append(os.path.join(image_base_path, filename))
            frame_number_ls.append(int(filename.split('_')[-1].split('.')[0]))

    # Sort the list by the last number before .jpg, do not use regular sort, because 10 will be before 2
    originalImagePaths = [x for _,x in sorted(zip(frame_number_ls,originalImagePaths))]
    print("originalImagePaths: ", originalImagePaths)

    # Also return the manual image path
    manual_image_dir = os.path.join(os.getcwd(), '..', 'public/dataset/manual_img' , selected_category, selected_subcategory, selected_step)
    if os.path.exists(manual_image_dir):
        if len(os.listdir(manual_image_dir)) == 1:
            print ("manual image exists")
            manualImagePath = os.path.join(manual_image_dir, os.listdir(manual_image_dir)[0]).split('public/')[-1]
        elif len(os.listdir(manual_image_dir)) > 1:
            print ("manual image exists, but more than one")
        
    
    if "manualImagePath" not in locals():
        manualImagePath = originalImagePaths[0]

    return jsonify({"originalImagePaths": originalImagePaths, "manualImagePath": manualImagePath})


#################### Projection Modification ####################
def rotate(axis, degree):
    if axis == 'x':
        return [[1, 0, 0],
                [0, math.cos(math.radians(degree)), -math.sin(math.radians(degree))],
                [0, math.sin(math.radians(degree)), math.cos(math.radians(degree))]]
    elif axis == 'y':
        return [[math.cos(math.radians(degree)), 0, math.sin(math.radians(degree))],
                [0, 1, 0],
                [-math.sin(math.radians(degree)), 0, math.cos(math.radians(degree))]]
    elif axis == 'z':
        return [[math.cos(math.radians(degree)), -math.sin(math.radians(degree)), 0],
                [math.sin(math.radians(degree)), math.cos(math.radians(degree)), 0],
                [0, 0, 1]]
    
@app.route('/change-current-projection', methods=['POST'])
def change_current_projection():
    data = request.json
    selected_axis = data["Input1"]
    angle_or_distance= float(data["Input2"])
    rotation_or_translation = data["Input3"]

    prev_pose_data = json.load(open(POSE_ESTIMATION_DATA_PATH))
    ext_mat = np.array(prev_pose_data["extrinsic"])
    if rotation_or_translation == 'r':
        rotation_matrix = rotate(selected_axis, angle_or_distance)
        ext_mat[:3, :3] = np.dot(ext_mat[:3, :3], rotation_matrix)
    elif rotation_or_translation == 't':
        if selected_axis == 'x':
            ext_mat[0, 3] += angle_or_distance
        elif selected_axis == 'y':
            ext_mat[1, 3] += angle_or_distance
        elif selected_axis == 'z':
            ext_mat[2, 3] += angle_or_distance
    prev_pose_data["extrinsic"] = ext_mat.tolist()

    with open(POSE_ESTIMATION_DATA_PATH, 'w') as outfile:
        json.dump(prev_pose_data, outfile, indent=4)

    # Plot diagram
    cmd_to_execute = "python ./project_part.py --json " + POSE_ESTIMATION_DATA_PATH 
    conda_cmd = f'conda run -n IKEA-dataset {cmd_to_execute}'
    ret = subprocess.run(conda_cmd, shell=True, capture_output=True, text=True)
    
    print("Re-projection finished")
    return_code = ret.returncode
    print(ret)
    print(return_code)
    

    return jsonify({"success": True})


#################### Save Functions ####################

def handle_frame(frame, frame_num, parts, ext_mat=None, int_mat=None, new_mask=None, segmentation_data=None, pose_estimation_data=None):
    frame_exist = False
    parts_exist = False
    if int(frame['frame_id']) == int(frame_num):
        frame_exist = True
        # Check if parts already exists
        for i in range(len(frame['parts'])):
            if is_same_parts(frame['parts'][i],parts):
                parts_exist = True
                if ext_mat is not None and int_mat is not None: # Update extrinsic and intrinsic
                    frame, data_exist = update_ext_int(frame, i, ext_mat, int_mat, pose_estimation_data)
                if new_mask is not None and segmentation_data is not None:# Update mask
                    frame, data_exist = update_mask(frame, i, new_mask, segmentation_data)
    return frame_exist, parts_exist, data_exist

def update_ext_int(frame, index, ext_mat, int_mat, pose_estimation_data):
    new_frame = frame.copy()
    data_exist = True
    assert len(new_frame['extrinsics']) == len(new_frame['intrinsics'])
    if new_frame['extrinsics'] == [] or new_frame['intrinsics'] == []:  
        new_frame['extrinsics'] = [[]] * len(new_frame['parts'])
        new_frame['intrinsics'] = [[]] * len(new_frame['parts'])
        data_exist = False

    new_frame['extrinsics'][index] = ext_mat
    new_frame['intrinsics'][index] = int_mat
    old_image_path = pose_estimation_data['output_path'][:-8] # Before _new.jpg
    new_image_path = pose_estimation_data['output_path']

    remove_and_rename(old_image_path, new_image_path, '')
    return new_frame, data_exist

def update_mask(frame, index, new_mask, segmentation_data):
    new_frame = frame.copy()
    data_exist = True
    if new_frame['mask'] == []:  
        new_frame['mask'] = [{}] * len(new_frame['parts'])
        data_exist = False


    new_frame['mask'][index] = new_mask
    old_image_path = segmentation_data['output_path'][:-8] # Before _new.jpg
    new_image_path = segmentation_data['output_path']
    remove_and_rename(old_image_path, new_image_path, '')
    remove_and_rename(old_image_path, new_image_path, '_with_frame_img')
    remove_and_rename(old_image_path, new_image_path, '_with_points')

    return new_frame, data_exist

def remove_and_rename(old_image_path, new_image_path, image_type):
    new_image_path = new_image_path[:-4] + image_type + '.jpg'
    try:
        os.remove(old_image_path + image_type + '.jpg')
    except:
        print("Old image" + old_image_path + image_type + ".jpg not found")
    else:
        print("removing :", old_image_path + image_type + '.jpg')

    try:
        os.rename(new_image_path, old_image_path + image_type + '.jpg')
    except:
        print("New image" + new_image_path + " not found")
    else:
        print("rename: ", new_image_path.split('/')[-1], old_image_path.split('/')[-1] + image_type + '.jpg')



@app.route('/save-mask', methods=['POST'])
def save_mask():
    segmentation_data = get_data(SEGMENTATION_DATA_PATH)
    parts = segmentation_data["parts"]
    category = segmentation_data["category"]
    name = segmentation_data["name"]
    step_id = segmentation_data["step_id"]
    frame_num = segmentation_data["frame_id"]
    video_id = segmentation_data["video_id"]

    json_data = get_data(DATA_JSON_PATH)
    video = get_video(json_data, category, name, step_id, video_id)

    if video is not None:
        for frame in video['frames']:
            frame_exist, parts_exist, data_exist = handle_frame(frame, frame_num, parts, new_mask=segmentation_data['new_mask'], segmentation_data=segmentation_data)
    write_data(json_data)
    if not frame_exist:
        return jsonify({"messages": ["Frame does not exist, write new frame, mask saved!"]}), 200
    elif not parts_exist or not data_exist:
        return jsonify({"messages": ["Frame exist, but parts data does not exist, write new parts, mask saved!"]}), 200
    else:
        return jsonify({"messages": ["Parts already exist, update mask!"]}), 200
    

@app.route('/save-projection', methods=['POST'])
def save_projection():
    pose_estimation_data = get_data(POSE_ESTIMATION_DATA_PATH)
    ext_mat = pose_estimation_data["extrinsic"]
    int_mat = pose_estimation_data["intrinsic"]
    parts = pose_estimation_data["part_idxs"]
    category = pose_estimation_data["category"]
    name = pose_estimation_data["name"]
    step_id = pose_estimation_data["step_id"]
    frame_num = pose_estimation_data["frame_id"]
    video_id = pose_estimation_data["video_id"]

    json_data = get_data(DATA_JSON_PATH)
    video = get_video(json_data, category, name, step_id, video_id)

    if video is not None:
        for frame in video['frames']:
            frame_exist, parts_exist, data_exist = handle_frame(frame, frame_num, parts, ext_mat=ext_mat, int_mat=int_mat, pose_estimation_data=pose_estimation_data)

    write_data(json_data)
    if not frame_exist:
        return jsonify({"messages": ["Frame does not exist, write new frame, projection saved!"]}), 200
    elif not parts_exist or not data_exist:
        return jsonify({"messages": ["Frame exist, but parts data does not exist, write new parts, projection saved!"]}), 200
    else:
        return jsonify({"messages": ["Parts already exist, update projection!"]}), 200



if __name__ == '__main__':
    app.run(port=5000)

----------------------------------------
-------./backend/SAM/sam.py-------
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from segment_anything.modeling import Sam
import json
import argparse
import cv2
import numpy as np
import random
from pycocotools import mask as mask_utils
from typing import Optional, Tuple



class ExtendedSamPredictor(SamPredictor):
    def __init__(self, sam_model: Sam) -> None:
        super().__init__(sam_model=sam_model)

    def load_image_embedding(self, embedding: np.ndarray, original_size: Tuple[int, int], input_size: Tuple[int, int]) -> None:
        """
        Loads the image embedding from a previously calculated result.
        
        Arguments:
          embedding (np.ndarray): The previously calculated image embedding.
          original_size (tuple(int, int)): The size of the original image, in (H, W) format.
          input_size (tuple(int, int)): The size of the transformed image, in (H, W) format.
        """
        # Convert numpy array to torch tensor
        embedding_torch = torch.from_numpy(embedding).to(self.device)
        
        # Set the features of the model to the loaded embedding
        self.features = embedding_torch

        # Set original and input sizes
        self.original_size = original_size
        self.input_size = input_size
        
        # Since an image embedding is set, we update the flag
        self.is_image_set = True


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)



'''
The decode_rle_to_mask function is used to decode the RLE format mask into a binary mask. 
The get_boundary_and_inner_points function is used to obtain the boundary points and inner points of the mask. 
The num_inner_points parameter is the number of points you want to randomly draw from inside the mask. 
If the number of points inside the mask is less than num_inner_points, then all interior points are returned.

>>>  mask = decode_rle_to_mask(rle, h, w)
>>> boundary_points, inner_points = get_boundary_and_inner_points(mask)

'''


def get_inner_points(mask, num_inner_points=2):
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Convert contours to points, just need a few points
    boundary_points = []
    for contour in contours:
        for point in contour:
            boundary_points.append(point[0])
    
    # boundary_points = random.sample(boundary_points, 1)
    
    # Get inner points
    y_indices, x_indices = np.where(mask == 1)
    indices = list(zip(x_indices, y_indices))
    
    if len(indices) > num_inner_points:
        # Get random sample inside the mask
        inner_points = random.sample(indices, num_inner_points)

    else:
        inner_points = indices
    
    return np.array(inner_points)

def remove_mask_overlap(previous_masks, binary_mask):
    for prev_mask in previous_masks:
        if prev_mask == {}:
            continue
        else:
            rle = {
                'counts': prev_mask['counts'].encode('ascii'),
                'size': prev_mask['size'],
            }

            mask = mask_utils.decode(prev_mask)
            binary_mask[mask == 1] = 0
        return binary_mask

if __name__ == "__main__":

    HOME = os.getcwd() + "/SAM"
    # HOME = os.getcwd()
    CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MODEL_TYPE = "vit_h"
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)


    # Use argparse to get the input

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--json_path', type=str, help='Path to the json file', default='./segmentation_data.json' )
    args = parser.parse_args()

    # Load the json file
    with open(args.json_path, 'r') as f:
        data = json.load(f)

    image_path = data['image_path']
    ouptut_path = data['output_path']
    input_point = np.array(eval(data['positive_points']) + eval(data['negative_points']))
    input_label = np.array([1] * len(eval(data['positive_points'])) + [0] * len(eval(data['negative_points'])))
    image_embedding_path = data['image_embedding_path']
    print(input_point)
    print(input_label)


    image_bgr = cv2.imread(image_path)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    original_image = image

    sam.to(device=DEVICE)
    image_embedding = np.load(image_embedding_path)

    # Create a new predictor
    new_predictor = ExtendedSamPredictor(sam)

    # Get the original size
    original_size = original_image.shape[:2]  # Height and width of the original image

    # Apply the transform to get the input size
    input_image = new_predictor.transform.apply_image(original_image)
    input_size = input_image.shape[:2]  # Input size after transformation

    # Load the saved embedding and sizes
    new_predictor.load_image_embedding(image_embedding, original_size, input_size)


    
    
    
    # # predictor = SamPredictor(sam)
    # predictor = ExtendedSamPredictor(sam)
    # # predictor.set_image(image)
    # image_embedding = np.load(image_embedding_path)
    # predictor.load_image_embedding(image_embedding, original_size, input_size)
    



    # The points coordinates are in the range [0, 1], and should be rescaled to the image size
    h, w = image.shape[:2]
    input_point[:, 0] *= w
    input_point[:, 1] *= h

    if data['previous_mask_exist']:
        previous_masks = data['previous_masks']


        # Convert the previous mask to negative points
        negative_points_from_previous_mask = []
        for prev_mask in previous_masks['masks']:
            mask = mask_utils.decode(prev_mask, h, w)
            boundary_points, inner_points = get_inner_points(mask)
            negative_points_from_previous_mask.append(inner_points)
            negative_points_from_previous_mask.append(boundary_points)
        negative_points_from_previous_mask = np.concatenate(negative_points_from_previous_mask, axis=0)

        # Add the negative points from the previous mask to the input points
        input_point = np.concatenate([input_point, negative_points_from_previous_mask], axis=0)
        input_label = np.concatenate([input_label, np.zeros(len(negative_points_from_previous_mask))], axis=0)
                            


    masks, scores, logits = new_predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(masks[0], plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.gca().set_position([0, 0, 1, 1])

    # Remove the axis   
    plt.axis('off')
    # Save the figure
    # Make directory if it doesn't exist
    output_dir = os.path.join(HOME, '/'.join(ouptut_path.split("/")[:-1]))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(os.path.join(HOME, ouptut_path), bbox_inches='tight', pad_inches=0)
    plt.close()

    # Save the mask to a json file using RLE encoding to the input json
    rle = mask_utils.encode(np.asfortranarray(masks[0].astype(np.uint8)))
    rle['counts'] = rle['counts'].decode('ascii')
    data['new_mask']={"size": [masks[0].shape[0], masks[0].shape[1]], "counts": rle['counts']}
    with open(args.json_path, 'w') as f:
        json.dump(data, f, indent=4)


# Example of how to use the SAM model:
# python sam.py --image_path ../../public/sample_frame_images/T4ijaGT1eaM_cut_2.png --output_path ./output_images --positive_points "[[450, 375]]" --negative_points "[]"

----------------------------------------
-------./backend/utils/data_utils.py-------
import json

def get_video(json_data, category, name, step_id, video_id):
    for data in json_data:
        if data['category'] == category and data['name'] == name:
            for video in data['steps'][int(step_id.split('_')[-1])]['video']:
                if video['video_id'].split('watch?v=')[-1] == video_id:
                    return video
    return None


def get_data(data_path):
    try:
        with open(data_path) as json_file:
            data = json.load(json_file)
        return data
    except FileNotFoundError:
        print(f"File {data_path} not found.")
        return None

def write_data(json_data):
    with open('new_data.json', 'w') as outfile:
        json.dump(json_data, outfile , indent=4)

def is_same_parts(parts1, parts2):
    # "0,3,4" and "3,4,0" are same parts
    parts1 = parts1.split(',')
    parts2 = parts2.split(',')
    if len(parts1) != len(parts2):
        return False
    for part in parts1:
        if part not in parts2:
            return False
    return True



----------------------------------------
-------./src/index.tsx-------
import React from "react"
import ReactDOM from "react-dom/client"
import "./index.css"
import App from "./App"

const root = ReactDOM.createRoot(document.getElementById("root") as HTMLElement)
root.render(
	<React.StrictMode>
		<App />
	</React.StrictMode>
)


----------------------------------------
-------./src/App.tsx-------
import styled from "styled-components"
import { IStyled } from "./type"
import { Scene } from "./component/Scene"
import { useState } from "react"
// import { numberOfModel } from "./component/ModelLoader"
// import { models } from "./component/ModelLoader"
import { useEffect } from "react";
import { log } from "console"
import { ModelLoader } from "./component/ModelLoader"
import { SwitchModeButton, ProcessCoordinatesButton, OtherButton_Red, OtherButton_Green, OtherButton_Grey } from "./component/Button"

const host = "localhost"
const port = 5000




// Define the interface for the coordinates
interface Coordinate {
	x: number;
	y: number;
  }
  
// Make a list to store the coordinates
// const CoordLs2D: Coordinate[] = [];



//Add a state variable to keep track of the returned image

  
interface AppProps extends IStyled {}

const RawApp = (props: AppProps) => {
	// Add a state variable to keep track of the returned image
	const [returnedProjectImagePaths, setReturnedProjectImagePaths] = useState<string[]>([]);
	// Add a state variable to keep track of the returned image
	const [returnedSegmentationImagePaths, setReturnedSegmentationImagePaths] = useState<string[]>([]);
	// Set up a state variable for the button toggled state
	const [Projection, setProjection] = useState(false); // Two mode: Projection and Segmentation
	const [coordinates3D, setCoordinates3D] = useState<THREE.Vector3[]>([]);
	const [mode, setMode] = useState('Add'); // 'Add' or 'Remove'
	const [positiveKeypoints, setPositiveKeypoints] = useState<Coordinate[]>([]);
	const [negativeKeypoints, setNegativeKeypoints] = useState<Coordinate[]>([]);
	const [originalImagePath, setOriginalImagePath] = useState('');
	const [frames, setFrames] = useState([]);

	const [currentImage, setCurrentImage] = useState(originalImagePath);
	const [currentUpperImage, setCurrentUpperImage] = useState(originalImagePath);

	const [Categories, setCategories] = useState([]);
	const [selectedCategory, setSelectedCategory] = useState('');
	const [selectedSubCategory, setSelectedSubCategory] = useState('');
	const [subCategories, setSubCategories] = useState([]);
	const [selectedStep, setSelectedObject] = useState('');
	const [objects, setObjects] = useState([]);
	const [selectedFrame, setSelectedFrame] = useState('');
	// const [frames, setFrames] = useState([]);
	const [modelFilePaths, setModelFilePaths] = useState<string[]>([]);
	const [currentModelFilePaths, setCurrentModelFilePaths] = useState<string[]>([]);
	const [checkBoxStatus, setCheckBoxStatus] = useState<Record<string, boolean>>({});
	const[selectedProjctionImage, setSelectedProjectionImage] = useState('');
	const [isClean3DClicked, setIsClean3DClicked] = useState(false);
	const [CoordLs2D, setCoordLs2D] = useState<Coordinate[]>([]);
	const [ManualImagePath, setManualImagePath] = useState('');
	const [input1, setInput1] = useState('');
	const [input2, setInput2] = useState('');
	const [input3, setInput3] = useState('');
	const [selectedSegmentationImage, setSelectedSegmentationImage] = useState('');
	const [messages, setMessages] = useState([]);


	

	




	
	const handleToggleButtonClick = () => {
		// Update upper image (opposite of current image)
		if(!Projection){
			const unique_suffix = "?t=" + new Date().getTime();
			setCurrentUpperImage(selectedProjctionImage + unique_suffix);
		  } else {
			const unique_suffix = "?t=" + new Date().getTime();
			setCurrentUpperImage(selectedSegmentationImage + unique_suffix);
		  }

		setProjection(!Projection);


		const color = Projection ? 'green' : 'red';
		const text = Projection ? 'ON' : 'OFF';
	
		// Send dataToSend to backend
	
		// Change the button text and color
		const buttonElement = document.getElementById('toggleButton');
		if (buttonElement) {
			buttonElement.style.backgroundColor = color;
			buttonElement.innerHTML = text;
		}



		
	}

	
	const handleImageClick = (event: React.MouseEvent<HTMLImageElement>) => {

		//////////////////Display the 2D coordinates of the clicked point///////////////////
		// Get the coordinates of the click on the image
		const x = event.nativeEvent.offsetX;
		const y = event.nativeEvent.offsetY;
		console.log(`Image clicked at coordinates (${x}, ${y})`);
		
		let color = 'lightgreen';
		// Rescale the coordinates to 0-1 range
		const x_norm = x / event.currentTarget.width;
		const y_norm = y / event.currentTarget.height;
		console.log(`Normalized coordinates: (${x_norm}, ${y_norm})`);
		const coord: Coordinate = { x: x_norm, y: y_norm };


		if (Projection){
			// Add the coordinates to the list
			setCurrentImage(originalImagePath)
			setCoordLs2D([...CoordLs2D, coord]);
			console.log("CoordLs2D: " );
			console.log(CoordLs2D);
			
		
		}else{

			if(mode === 'Add'){
				setPositiveKeypoints([...positiveKeypoints, coord]);
				console.log("positiveKeypoints: " );
				console.log(positiveKeypoints);

			}else{
				setNegativeKeypoints([...negativeKeypoints, coord]);
				console.log("negativeKeypoints: " );
				console.log(negativeKeypoints);
				// Set color to red
				color = 'red';
			}

		}

			// Create a new div
			const pointDiv = document.createElement("div");
			// Style the div to look like a dot
			pointDiv.style.position = "absolute";
			pointDiv.style.top = (y - 5) + "px"; // subtracting half the size to center the dot
			pointDiv.style.left = (x - 5) + "px"; // subtracting half the size to center the dot
			pointDiv.style.width = "10px";
			pointDiv.style.height = "10px";
			pointDiv.style.backgroundColor = color;
			pointDiv.style.borderRadius = "50%";
		
			// Append the div to the image container
			const imageContainer = document.getElementById("image-container");
			if (imageContainer) {
				imageContainer.appendChild(pointDiv);
			} else {
				console.error("Image container not found");
			}

	};

	const handleClean = () => {
		// TODO: Clean the list of 3D coordinates

		setPositiveKeypoints([]);
		setNegativeKeypoints([]);
		setCoordLs2D([]);
		// Remove all dots from the image container, except the first one
		const imageContainer = document.getElementById("image-container");
		if (imageContainer) {
			while (imageContainer.children.length > 1) {
				imageContainer.removeChild(imageContainer.lastChild as Node);
			}
		} else {
			console.error("Image container not found");
		}

	
	  };
	const handleClean3D = () => {
		setIsClean3DClicked(true);

		console.log('Button Clean 3D clicked in App.tsx');

		setCoordinates3D([]);
	};
	
	const handleUndo = () => {
		if (Projection){
			// Remove the last element from the list
			setCoordLs2D(CoordLs2D.slice(0, -1));
		}else{
			if(mode === 'Add'){
				// Remove the last element from the list
				positiveKeypoints.pop();

			}else{
				// Remove the last element from the list
				negativeKeypoints.pop();
				
			}
		}
		// Remove the last dot from the image container
		const imageContainer = document.getElementById("image-container");
		if (imageContainer && imageContainer.children.length > 1) {
			imageContainer.removeChild(imageContainer.lastChild as Node);
		} else {
			console.error("Image container not found");
		}
	};


	function handleProcessCoordinates() {
		setMessages([]);
		if (Projection){ // Pass both 2D and 3 D coordinates
			

		// Send coordinates to backend
		fetch('http://'+ host+':'+port+'/pose-estimation', {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json'
			},
			body: JSON.stringify({"2d-coordinates": CoordLs2D, "3d-coordinates": coordinates3D
									,"image-path": currentImage, "Category": selectedCategory, "SubCategory": selectedSubCategory, "Object": selectedStep,
									'currentModelFilePaths': currentModelFilePaths })
			
		})
		.then(response => {
			console.log("Response received:", response);
			return response.json();
		})
		.then(data => {

			console.log("Data received:", data);
			console.log(data);
			if (data.error){
				console.log("Error received:", data);
				alert(data.error);
			}else{
				if (data.messages){
					setMessages(data.messages);
				}

				const unique_suffix = "?t=" + new Date().getTime();
				setCurrentUpperImage(data.imagePath + unique_suffix);
			
				fetch('http://'+host + ':' + port + '/get-segmentation-image-path-initial', {
					method: 'POST',
					headers: {
						'Content-Type': 'application/json'
					},
					body: JSON.stringify({'Category': selectedCategory, 'SubCategory': selectedSubCategory, 'Object': selectedStep, 'Frame': selectedFrame, 'OriginalImagePath': originalImagePath})
				})
				.then(response => response.json())
				.then(data => {
					if (data.error){
						console.log("Error received:", data);
						alert(data.error);
					}
					if (data.messages){
						setMessages(data.messages);
					}
					console.log(data);
					// Append a timestamp to the image URLs
					setReturnedProjectImagePaths(data.projectImagePaths);
					console.log("project image paths: ");
					console.log(data.projectImagePaths);
					setReturnedSegmentationImagePaths(data.segmentationImagePaths);
					console.log("segmentation image path: ");
					console.log(data.segmentationImagePath);
					console.log("current upper image: ");
					console.log(currentUpperImage);
				}
		
				
				)
				.catch(error => console.error('Error:', error));

				}
			})
		.catch(error => console.error('Error:', error));

		}else{ // Pass only positive and negative keypoints, and display the segmentation result at the original image
		fetch('http://'+ host+':'+port+'/segmentation', {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json'
			},
			body: JSON.stringify({"positive-keypoints": positiveKeypoints, "negative-keypoints": negativeKeypoints, "image-path": originalImagePath,
									"Category": selectedCategory, "SubCategory": selectedSubCategory, "Object": selectedStep, 'currentModelFilePaths': currentModelFilePaths })

		})
		.then(response => {
			return response.json();
		}
		)
		.then(data => {
			if (data.error){
				console.log("Error received:", data);
				alert(data.error);
			}
			if (data.messages){
				setMessages(data.messages);
			}
			console.log("Data received:", data);
			console.log(data);
			if (data.error){
				console.log("Error received:", data);
				alert(data.error);
			}else{
				const unique_suffix = "?t=" + new Date().getTime();
				setCurrentUpperImage(data.imagePath + unique_suffix);
			
				fetch('http://'+host + ':' + port + '/get-segmentation-image-path-initial', {
					method: 'POST',
					headers: {
						'Content-Type': 'application/json'
					},
					body: JSON.stringify({'Category': selectedCategory, 'SubCategory': selectedSubCategory, 'Object': selectedStep, 'Frame': selectedFrame, 'OriginalImagePath': originalImagePath})
				})
				.then(response => response.json())
				.then(data => {
				if (data.error){
						console.log("Error received:", data);
						alert(data.error);
				}
				if (data.messages){
					setMessages(data.messages);
				}
										
				console.log(data);
				// Append a timestamp to the image URLs
				const unique_suffix = "?t=" + new Date().getTime();
				for (let i = 0; i < data.projectImagePaths.length; i++){
					data.projectImagePaths[i] = data.projectImagePaths[i] + unique_suffix;
				}
				for (let i = 0; i < data.segmentationImagePaths.length; i++){
					data.segmentationImagePaths[i] = data.segmentationImagePaths[i] + unique_suffix;
				}
				
				setReturnedProjectImagePaths(data.projectImagePaths);
				console.log("project image paths: ");
				console.log(data.projectImagePaths);
				setReturnedSegmentationImagePaths(data.segmentationImagePaths);
				console.log("segmentation image path: ");
				console.log(data.segmentationImagePath);
				console.log("current upper image: ");
				console.log(currentUpperImage);
				}
		
				
				)
				.catch(error => console.error('Error:', error));

				}
			}
		)
		.catch(error => {
			console.error('Error:', error);
			alert(error);
		} );

	}
	}

	
	const handle3DCoordinates = (newCoordinates: THREE.Vector3) => {
		console.log({"coordinates3D before update: ": coordinates3D});
		setCoordinates3D([...coordinates3D, newCoordinates]);


		console.log({"coordinates3D after update: ": coordinates3D});
	}

	// Handle save mask button click
	const handleSaveMask = () => {
		// Send save mask request to backend
		fetch('http://'+host + ':' + port + '/save-mask', {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json'
			},
			// body: JSON.stringify({ "mask": mask })
		})
		.then(response => response.json())
		.then(data => {
			console.log(data);
			// Handle response data here
			if (data.error){
				console.log("Error received:", data);
				alert(data.error);
			}
			if (data.messages){
				setMessages(data.messages);
			}
		}
		)
		.catch(error => console.error('Error:', error));
	}

	// Set category at the web load
	useEffect(() => {
		fetch('http://'+host + ':' + port + '/get-categories')
			.then(response => response.json())
			.then(data => {
				if (data.error){
					console.log("Error received:", data);
					alert(data.error);
				}
				if (data.messages){
					setMessages(data.messages);
				}
				console.log(data);
				setCategories(data.categories);
			})
			.catch(error => console.error('Error:', error));
	}
	, []);


	const handleCategoryChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
		setSelectedCategory(event.target.value);
		setSelectedSubCategory("");
		setSelectedObject("");
		setSubCategories([]);
		setObjects([]);
		setFrames([]);

		if (event.target.value !== "") {
			fetch('http://'+host + ':' + port + '/get-subcategories', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify({ category: event.target.value })
			})
			.then(response => response.json())
			.then(data => {
				if (data.error){
					console.log("Error received:", data);
					alert(data.error);
				}
				if (data.messages){
					setMessages(data.messages);
				}
				console.log(data);
				setSubCategories(data.subcategories);

			}
			)
			.catch(error => console.error('Error:', error));
		}
	}

	const handleSubCategoryChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
		setMessages([]);
		setSelectedSubCategory(event.target.value);
		setSelectedObject("");
		setObjects([]);

		// Initialize check box status for each file path
		const initialStatus: Record<string, boolean> = {};
		modelFilePaths.forEach((path: string) => {
			initialStatus[path] = false;
		});
		setCheckBoxStatus(initialStatus);

		if (event.target.value !== "") {
			fetch('http://'+host + ':' + port + '/get-steps', {

				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify({ category: selectedCategory, subCategory: event.target.value })
			})
			.then(response => response.json())
			.then(data => {
				if (data.error){
					console.log("Error received:", data);
					alert(data.error);
				}
				if (data.messages){
					setMessages(data.messages);
				}
				console.log(data);
				setObjects(data.objects);
				setModelFilePaths(data.filePaths);

				const initialStatus: Record<string, boolean> = {};
				data.filePaths.forEach((path: string) => {
				  initialStatus[path] = false;
										});			
			  setCheckBoxStatus(initialStatus);

			}
			)
			.catch(error => console.error('Error:', error));
		}
	}

	const handleStepChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
		setMessages([]);
		setSelectedObject(event.target.value);
		if (event.target.value !== "") {
			fetch('http://'+host + ':' + port + '/get-object', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify({ category: selectedCategory, subCategory: selectedSubCategory, object: event.target.value })
			})
			.then(response => response.json())
			.then(data => {
				if (data.error){
					console.log("Error received:", data);
					alert(data.error);
				}
				if (data.messages){
					setMessages(data.messages);
				}
				console.log(data);
				// if (data.length > 0) {
				  setFrames(data.originalImagePaths);
				  console.log("original image paths: ");
				  console.log(data.originalImagePaths);
				  setManualImagePath(data.manualImagePath);
				  console.log("manual image path: ");
				  console.log(data.manualImagePath);

				  
			  })
			
			.catch(error => console.error('Error:', error));
		}
	}

	const handleFrameChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
		setMessages([]);

		setSelectedFrame(event.target.value); // get the last element of the path
		setCurrentImage(event.target.value);
		setOriginalImagePath(event.target.value);
		fetch('http://'+host + ':' + port + '/get-segmentation-image-path-initial', {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json'
			},
			body: JSON.stringify({'Category': selectedCategory, 'SubCategory': selectedSubCategory, 'Object': selectedStep, 'Frame': event.target.value, 'OriginalImagePath': event.target.value})
		})
		.then(response => response.json())
		.then(data => {
			if (data.error){
				console.log("Error received:", data);
				alert(data.error);
			}
			if (data.messages){
				setMessages(data.messages);
			}
		  console.log(data);
		  // Append a timestamp to the image URLs
		  setReturnedProjectImagePaths(data.projectImagePaths);
		  console.log("project image paths: ");
		  console.log(data.projectImagePaths);
		  setReturnedSegmentationImagePaths(data.segmentationImagePaths);
		  console.log("segmentation image path: ");
		  console.log(data.segmentationImagePaths);
		  console.log("current upper image: ");
		  console.log(currentUpperImage);
		}

		
		)
		.catch(error => console.error('Error:', error));

	}

	const [opacity, setOpacity] = useState(1);
	  
	const handleCheckBoxChange = (path: string) => {
		const newCheckBoxStatus = { ...checkBoxStatus };
		newCheckBoxStatus[path] = !checkBoxStatus[path];
		setCheckBoxStatus(newCheckBoxStatus);
		console.log("check box status: ");
		console.log(newCheckBoxStatus);

		// Update the model file paths
		const newModelFilePaths = modelFilePaths.filter((path: string) => {
			return newCheckBoxStatus[path];
		}
		);
		setCurrentModelFilePaths(newModelFilePaths);
		console.log("current model file paths: ");
		console.log(newModelFilePaths);
		
	};

	const handleSaveProjection = () => {
		setMessages([]);
		fetch('http://'+host + ':' + port + '/save-projection', {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json'
			},
			body: JSON.stringify({'Category': selectedCategory, 'SubCategory': selectedSubCategory, 'Object': selectedStep, 'Frame': selectedFrame, 'OriginalImagePath': originalImagePath,
			'currentModelFilePaths': currentModelFilePaths})
		})
		.then(response => response.json())
		.then(data => {
			if (data.error){
				console.log("Error received:", data);
				alert(data.error);
			}
			if (data.messages){
				setMessages(data.messages);
			}
			console.log(data);

		}				
		)
		.catch(error => console.error('Error:', error));
	}


	const handleProjctionImage = (event: React.ChangeEvent<HTMLSelectElement>) => {
		setSelectedProjectionImage(event.target.value);
		console.log("selected projection image: ");
		console.log(event.target.value);
		const unique_suffix = "?t=" + new Date().getTime();
		setCurrentUpperImage(event.target.value + unique_suffix);
	}

	const handleSegmentationImage = (event: React.ChangeEvent<HTMLSelectElement>) => {
		setSelectedSegmentationImage(event.target.value);
		console.log("selected projection image: ");
		console.log(event.target.value);
		const unique_suffix = "?t=" + new Date().getTime();
		setCurrentUpperImage(event.target.value + unique_suffix);
	}

	const handleInput1Change = (e : React.ChangeEvent<HTMLInputElement>) => {
		setInput1(e.target.value);
	  };
	
	  const handleInput2Change = (e : React.ChangeEvent<HTMLInputElement>) => {
		setInput2(e.target.value);
	  };
	
	  const handleInput3Change = (e : React.ChangeEvent<HTMLInputElement>) => {
		setInput3(e.target.value);
	  };
	
	  const handleSubmit = async () => {
		setMessages([]);
		fetch('http://'+host + ':' + port + '/change-current-projection', {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json'
			},
			body: JSON.stringify({'Input1': input1, 'Input2': input2, 'Input3': input3})
		})
		.then(response => response.json())
		.then(data => {
			if (data.error){
				console.log("Error received:", data);
				alert(data.error);
			}
			if (data.messages){
				setMessages(data.messages);
			}
			console.log(data);
		}
		)
		.catch(error => console.error('Error:', error));


	  };
	


	
	
	
	return (
		<main className={props.className}>
			<div className="gallery-container">
				{/* ---------------------- Left Section ----------------- */}
				<div className="left-section">
					
					
					<select value={selectedCategory} onChange={handleCategoryChange}>
						<option value="">Select Category</option>
						{/* List categories here */}
						{Categories.map(category => (
							<option value={category}>{category}</option>
						))}
						</select>
					<br />
					<select value={selectedSubCategory} onChange={handleSubCategoryChange}>
						<option value="">Select Sub Category</option>
						{subCategories && subCategories.map(subCategory => (
							<option value={subCategory}>{subCategory}</option>
						))}
					</select>

					<br />
					<select value={selectedStep} onChange={handleStepChange}>
						<option value="">Select Step</option>
						{/* List objects here */}
						{objects.map(object => (
							<option value={object}>{object}</option>
						))}
					</select>

					<br />
					<select value={selectedFrame} onChange={handleFrameChange}>
						<option value="">Select Frame</option>
						{/* List frames here */}
						{frames.map(frame => (
							<option value={frame}>{frame}</option>
						))}
					</select>
					<br />
					{/* Add a list of checkboxes here for each model file path, split the path by '/' and get the last element */}
					{modelFilePaths.map(path => ( 
						<div>
							<input type="checkbox" checked={checkBoxStatus[path]} onChange={() => handleCheckBoxChange(path)} />
							<label>{path.split('/').pop()}</label>
						</div>
					))}
					<br />
					<div style={{ display: Projection ? 'block' : 'none' }}>
					<select value={selectedProjctionImage} onChange={handleProjctionImage}>
						<option value="">Select Projection Image</option>
						{/* Returned Projection image paths here */}
						{returnedProjectImagePaths && returnedProjectImagePaths.map(path => (
							<option value={path}>
							{(path && path.length > 0 ? path.split('/').pop() || "" : "").split('?t=')[0]}
						</option>
						))}
					</select>

					</div>

					<div style={{ display: Projection ? 'none' : 'block' }}>
					<select value={selectedSegmentationImage} onChange={handleSegmentationImage}>
						<option value="">Select Segmentation Image</option>
						{/* Returned Projection image paths here */}
						{returnedSegmentationImagePaths && returnedSegmentationImagePaths.map(path => (
							<option value={path}>
							{(path && path.length > 0 ? path.split('/').pop() || "" : "").split('?t=')[0]}
						</option>
						))}
					</select>

					</div>

					<SwitchModeButton  id= "projection-segmentation-selection" onClick={handleToggleButtonClick}>
						{Projection ? "Projection Mode" : "Segmentation Mode"}
					</SwitchModeButton>
					<br />
					{/* Display manual image */}
					<img src={ManualImagePath} alt="Manual" />
							




				</div>
				
				{/* ---------------------- Middle Section ----------------- */}
				<div className="middle-section">
						
				<br />
				<br />
				<div>
					{messages.map((message, index) => (
						<p key={index} style={{
							fontFamily: 'Arial, sans-serif',
							color: '#333',
							lineHeight: '1.6',
							textAlign: 'left'
						}}>
							{message}
						</p>
					))}
				</div>

				<div className="scene-container">
					<Scene handle3DCoordinates={handle3DCoordinates} model={<ModelLoader filePaths={currentModelFilePaths} />} mode= {Projection ? "Projection" : "Segmentation"}
					isClean3DClicked={isClean3DClicked} // Pass the isCleanClicked state as prop
					setIsClean3DClicked={setIsClean3DClicked} // Pass the setIsCleanClicked setter function as prop
				  />
				</div>

	
					<div className="button-wrapper">

						<ProcessCoordinatesButton className="process-button" id="process-coordinates" onClick={handleProcessCoordinates}>	
							Process Coordinates
						</ProcessCoordinatesButton>
					{/* Display save button if in project mode */}
					{Projection && <OtherButton_Green  id="3d-save-button" onClick={handleSaveProjection}>Save the Last Projection</OtherButton_Green>}
					<div style={{ display: Projection ? 'block' : 'none' }}>
						<OtherButton_Red  id = 'clean-btn-project' onClick={handleClean3D}>Clean 3D Coords</OtherButton_Red>
					</div>
					</div>
				</div>
				
				{/* ---------------------- Right Section ----------------- */}
				<div className="right-section">

					{/* <Scene setCoordinates3D={setCoordinates3D} model={model[index]} effectOnKeyDown={effectOnKeyDown} /> */}

					<div className="part" id="top-part">
						

						{/* Place the image of 3d projection */}
						<div id="image-container-output" style={{ position: "relative" }}>
							<img src={currentImage} alt="frame_image" onClick={handleImageClick} />
							<br />
								<div id="image-container-output" style={{ position: "absolute", top: 0, left: 0 }}>
									<img src={currentUpperImage} alt="upper-image" style={{ opacity: opacity }} />
									{/*  Add a slider to control the opacity */}


							</div>
						</div>
					</div>
	
					<div className="part" id="middle-part">
						{/* Place the image of the object here */}
						<div id="image-container" style={{ position: "relative" }}>
							<img src={currentImage} alt="frame_image" onClick={handleImageClick} />

						</div>
					</div>
	
					<div className="part" id="bottom-part">
						{/* <!-- Placeholder --> */}
						<br />
						<div style={{ display: Projection ? 'none' : 'block' }}>
							<span>Add</span>
							<input
								type="range"
								min="0"
								max="1"
								value={mode === 'Add' ? 0 : 1}
								onChange={(e) => setMode(e.target.value === '0' ? 'Add' : 'Remove')}
								style={{ width: '50px' }}
							/>
							<span>Remove</span>

							
							{/* Add some space between slider and button */}
							<br />
							<OtherButton_Green  id = 'undo-btn' onClick={handleUndo}>Undo</OtherButton_Green>
							
							<OtherButton_Green  id = 'save-btn' onClick={handleSaveMask}>Save Mask</OtherButton_Green>
							<OtherButton_Red id = 'clean-btn' onClick={handleClean}>Clean</OtherButton_Red>
						</div>
						<div style={{ display: Projection ? 'block' : 'none' }}>
						<OtherButton_Red  id = 'clean-btn-project-2d' onClick={handleClean}>Clean 2D Coords</OtherButton_Red>
						</div>
							
						

						<div id="opacity-slider">
										<span>Opacity</span>
										<input
											type="range"
											min="0"
											max="1"
											step="0.01"
											value={opacity}
											onChange={(e) => setOpacity(parseFloat(e.target.value))}
										/>
										<br />
										<br />
					{/* Add 3 text box  to get input from user, and add a button to send all the values to backend, only if projection mode is selected */}
					<div style={{ display: Projection ? 'block' : 'none' }}>

						<div className="App">
						<label> Adjust Current Projection </label>
						<br />
						<label>Select Axis: </label>
						<input type="text" value={input1} onChange={handleInput1Change} />
						<br />
						<label>Enter Angle or Distance: </label>
						<input type="text" value={input2} onChange={handleInput2Change} />
						<br />
						<label>Rotation or Translation (r/t): </label>
						<input type="text" value={input3} onChange={handleInput3Change} />
						<br />
						<OtherButton_Green  id = 'send-btn' onClick={handleSubmit}>Send</OtherButton_Green>
						</div>
					</div>


						<br />
						


							
							</div>

					</div>
				</div>
			</div>
		</main>
	);
	};
	

// Add this styling
const App = styled(RawApp)`
    display: flex;
    height: 100vh; // 100% of the viewport height

    .gallery-container {
        display: flex;
        height: 100%;
        width: 100%;
    }

    .left-section {
        width: 15%; // Decrease the width of the left section
        background-color: #f5f5f5;
        padding: 20px;
        display: flex;
        flex-direction: column; // Set the direction of the children to be vertical
    }

    // .left-section .directory {
    //     flex: 1; // Set the directory to take up 2 parts of the available space
    //     overflow-y: auto; // Add a scrollbar when the content is too long
    // }

    // .left-section .image-container {
    //     flex: 1; // Set the image container to take up 3 parts of the available space
    //     display: flex;
    //     flex-direction: column;
    // }

    // .left-section .image-container img {
    //     flex: 1;
    //     object-fit: cover;
    //     width: 100%;
    // }

    .middle-section {
        flex: 3; // Increase the width of the middle section
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }

    .scene-container {
		width: 100%;
        flex: 3;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .button-wrapper {
        display: flex;
        justify-content: space-around;
        padding: 20px;
    }

    .right-section {
        flex: 3; // Increase the width of the right section
        display: flex;
        flex-direction: column;
        background-color: #f5f5f5;
    }
	// .message {
	// 	font-family: Arial, sans-serif;
	// 	color: #333;
	// 	line-height: 1.6;
	// 	text-align: left;
	// }
	
	

    // .part {
    //     flex: 1;
    //     display: flex;
	// 	height: 33;
    //     // align-items: center;
    //     // justify-content: center;
    // }

	.projection-segmentation-selection {
		position: absolute;
		bottom: 0;
	}
	  
	

`;

export default App;

----------------------------------------
-------./src/index.css-------
/* force full-screen */
html,
body,
#root {
	height: 100%;
}

/* default, copied from tailwindcss */
* {
	margin: 0;
	padding: 0;
}

h1,
h2,
h3,
h4,
h5,
h6 {
	font-size: inherit;
	font-weight: inherit;
}

ol,
ul {
	list-style: none;
	margin: 0;
	padding: 0;
}

img,
svg,
video,
canvas,
audio,
iframe,
embed,
object {
	display: block;
	/* vertical-align: middle; */
}

*,
::before,
::after {
	border-width: 0;
	border-style: solid;
	/* border-color: theme("borderColor.DEFAULT", currentColor); */
}

.google-map * {
	border-style: none;
}

a {
	text-decoration: none;
}


.right-section .part {
    height: 33.33vh;
}

/* #top-part, #middle-part {
    position: relative;
} */
#middle-part {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    height: 100%; /* Make sure it takes full height */
}



#top-part img, #middle-part img {
    max-width: 100%;
    max-height: 100%;
    object-fit: cover;
}




/* .fancy-button-green {
    margin-top: 10px;
    padding: 10px 20px;
    background-color: #4CAF50;
    border: none;
    color: white;
    text-align: center;
    font-size: 16px;
    transition: all 0.3s;
    cursor: pointer;
    border-radius: 4px;
} */

/* .fancy-button-green:hover {
    background-color: #45a049;
} */

/* .fancy-button-red {
    margin-top: 10px;
    padding: 10px 20px;
    background-color: #f44336;
    border: none;
    color: white;
    text-align: center;
    font-size: 16px;
    transition: all 0.3s;
    cursor: pointer;
    border-radius: 4px;
}

.fancy-button-red:hover {
    background-color: #da190b;
} */


#toggleButton {
    background-color: red;
    color: white;
}




----------------------------------------
-------./src/component/ModelLoader.tsx-------
import React from 'react';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader';
import { useLoader } from '@react-three/fiber';

// Model component that takes the model file path as a prop
function Model({ filePath, ...props }: { filePath: string } & JSX.IntrinsicElements['group']) {
  const obj = useLoader(OBJLoader, filePath);
  useLoader.preload(OBJLoader, filePath);
  return (
    <group {...props} dispose={null}>
      <primitive object={obj} />
    </group>
  );
}

// Component that takes an array of file paths and creates Model components
export function ModelLoader({ filePaths }: { filePaths: string[] }) {
  return (
    <>
      {filePaths.map((filePath, index) => (
        <Model key={index} filePath={filePath} />
      ))}
    </>
  );
}




----------------------------------------
-------./src/component/Button.tsx-------
import styled from "styled-components"
import { IStyled } from "../type"

interface ButtonProps extends IStyled {
	effectOnClick: () => void
	width?: number
	height?: number
}

const RawButton = (props: ButtonProps) => {
	const { className, effectOnClick, children } = props
	return (
		<button className={className} onClick={effectOnClick}>
			{children}
		</button>
	)
}

const Button = styled(RawButton)`
	width: ${({ width }) => width ?? 40}px;
	height: ${({ height }) => height ?? 40}px;
	background-color: transparent;

	display: flex;
	flex-flow: row nowrap;
	justify-content: center;
	align-items: center;
	cursor: pointer;
`


export const SwitchModeButton = styled.button`
	color: #BF4F74;
	font-size: 1em;
	margin: 1em;
	padding: 0.25em 1em;
	border: 2px solid #BF4F74;
	border-radius: 3px;
	&: hover {
	color: #FFFFFF;
	background-color: #BF4F74;
	}
`;

export const ProcessCoordinatesButton = styled(SwitchModeButton)`
	color: tomato;
	border-color: tomato;
	&: hover {
	color: #FFFFFF;
	background-color: tomato;
	}
`;

export const OtherButton_Red = styled(SwitchModeButton)`
  color: #f44336;
  border-color: #f44336;
  &: hover {
	color: #FFFFFF;
	background-color: #f44336;
	}
`;

//   switch foreground and background color if hover
export const OtherButton_Green = styled(SwitchModeButton)`
  color: #4CAF50;
  border-color: #4CAF50;
  &: hover {
	color: #FFFFFF;
	background-color: #4CAF50;
	}
`;

export const OtherButton_Grey = styled(SwitchModeButton)`
  color: #555555;
  border-color: #555555;
  &: hover {
	color: #FFFFFF;
	background-color: #555555;
	}
`;

export { Button }


----------------------------------------
-------./src/component/Scene.tsx-------
import styled from "styled-components";
import { IStyled } from "../type";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls, PerspectiveCamera } from "@react-three/drei";
import { Suspense, useEffect } from "react";
import * as THREE from "three";


const host = "localhost";
const port = 5000;


interface SceneProps<T = HTMLElement> extends IStyled {
	model?: React.ReactNode;
	handle3DCoordinates: (coordinates: THREE.Vector3) => void; // pdate the SceneProps interface in Scene.tsx to accept the new prop
	// Get current mode from app.tsx (Projection or Segmentation), make sure it is updated
	mode: string;
	isClean3DClicked: boolean; // Add the isCleanClicked prop
  	setIsClean3DClicked: (isClean3DClicked: boolean) => void; // Add the setIsCleanClicked setter function


  }
  

const InnerScene = (props: SceneProps<HTMLDivElement>) => {
	// const [coordinates3D_ls, setCoordinates3D_ls] = useState<THREE.Vector3[]>([]);


	const { gl, scene, camera } = useThree();
	const raycaster = new THREE.Raycaster();
	const mouse = new THREE.Vector2();
  
	useEffect(() => {
		if (props.isClean3DClicked) {
			// Remove all the dots from the scene
			scene.children.forEach((child) => {
				if (child.type === "Mesh") {
					scene.remove(child);
				}
			});
			props.setIsClean3DClicked(false);
		}


	  const handleClick = (event: MouseEvent) => {
		// Make sure the user is in the Projection mode
		if (props.mode === "Segmentation") {
			console.log("Segmentation mode, no 3D coordinates")
			alert("Segmentation mode, no 3D coordinates")
			return;
		}
		const rect = gl.domElement.getBoundingClientRect();
		mouse.x = (event.clientX - rect.left) / rect.width * 2 - 1;
		mouse.y = -(event.clientY - rect.top) / rect.height * 2 + 1;
  
		raycaster.setFromCamera(mouse, camera);
  
		const intersects = raycaster.intersectObjects(scene.children, true);
  
		if (intersects.length > 0) {
		  const point = intersects[0].point;
		  console.log('3D Coordinates: ', point);


			// Display a dot where the user clicked
			const geometry = new THREE.SphereGeometry(0.01);
			// Color green
			const material = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
			const sphere = new THREE.Mesh(geometry, material);
			sphere.position.set(point.x, point.y, point.z);
			scene.add(sphere);

			// Send the coordinates to the server, append it to the setCoordinates3D_ls
			// const coordinates3D = {x: point.x, y: point.y, z: point.z};

			// setCoordinates3D_ls([...coordinates3D_ls, point]);
			// Append the new point to the list of coordinates
			// props.setCoordinates3D((prevCoordinates: THREE.Vector3[]) => [...prevCoordinates, point]);
			const newCoordinates = point // obtain new 3D coordinates here
			props.handle3DCoordinates(newCoordinates);
			console.log({"New Coord from Scence : ": newCoordinates})


			  


		}
	  };


		// const perspCamera = camera as THREE.PerspectiveCamera;


	
		// // Print the camera parameters
		// console.log("Intrinsic parameters:");
		// console.log("Focal length (in pixels):", perspCamera.getFocalLength());
		// console.log("Image center (in pixels):", perspCamera.position.x, perspCamera.position.y);
		// console.log("Aspect ratio:", perspCamera.aspect);
	
		// console.log("Extrinsic parameters:");
		// console.log("Camera position (world coordinates):", camera.position.x, camera.position.y, camera.position.z);
		// console.log("Camera rotation (world coordinates):", camera.rotation.x, camera.rotation.y, camera.rotation.z);
	

  
	  gl.domElement.addEventListener('click', handleClick);
	  return () => gl.domElement.removeEventListener('click', handleClick);
	}, [gl, camera, scene, props.mode, props.handle3DCoordinates, props.isClean3DClicked, props.setIsClean3DClicked]);
  
	return (
	  <>
		<ambientLight intensity={0.5} />
		<spotLight intensity={1} position={[5, 5, 5]} />
		<OrbitControls minDistance={0.1} maxDistance={200} maxPolarAngle={Math.PI} minPolarAngle = {-Math.PI} />
		<PerspectiveCamera makeDefault position={[10, 10, 5]} near={0.01} far={2000} />

		<Suspense fallback={<></>}>{props.model}</Suspense>
	  </>
	);
  };
  

const RawScene = (props: SceneProps<HTMLDivElement>) => {
  return (
    <Canvas tabIndex={0} >
      <InnerScene {...props} />
    </Canvas>
  );
};

const Scene = styled(RawScene)``;

export { Scene };


