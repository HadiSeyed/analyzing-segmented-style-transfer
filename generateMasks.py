
import json
import numpy as np

def generateMaskJSON(image):
    '''
    Generate masks using Segment Anything Model
    
    inputs
    image: image as NumPy array

    output
    jsondata: JSON data as a Python dictionary

    '''
    from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
    from segment_anything.build_sam import sam_model_registry

    model_type = "vit_h"  # or another model type like "vit_l", "vit_b"
    checkpoint_path = "sam_vit_h_4b8939.pth"

    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    
    # # Generate regular images(mask)
    # # Generate masks using the SAM
    # mask_generator = SamAutomaticMaskGenerator(sam)
    # masks = mask_generator.generate(image)

    # # Generate json files
    # # Check https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/scripts/amg.py#L12
    # # Convert masks to COCO RLE format
    generator = SamAutomaticMaskGenerator(sam, output_mode="coco_rle")
    jsondata = generator.generate(image)

    # Ensure the output has an "annotations" key
    if "annotations" not in jsondata:
        jsondata = {"annotations": jsondata}

    return jsondata

def saveMaskJSON(jsondata, jsonfilename):
    '''
    Save JSON data to a file

    inputs
    jsondata: JSON data as a Python dictionary
    jsonfilename: path to save JSON file
    '''
    with open(jsonfilename, "w") as f:
        json.dump(jsondata, f)

def loadMaskJSON(jsonfilename):
    '''
    Load a presaved JSON file with mask information

    inputs
    jsonfilename: path to JSON file

    output
    jsondata: JSON data as a Python dictionary
    '''
    # f = open(jsonfilename, "r")
    # jsondata = json.load(f)

    with open(jsonfilename, "r") as f:
        jsondata = json.load(f)
    
    return jsondata