# You may be able to use a lot of the code in my original utils library,
# but I recommend trying to implement them yourself first to practice.


import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2


import torch.nn as nn
import torch.nn.functional as F


def loadImage(filename, asTensor=False):
    '''
    Load an image from a file

    inputs
    filename: path to image file
    asTensor: if True, return image as PyTorch Tensor, else return as NumPy array

    output
    image: image as PyTorch Tensor or NumPy array
    '''
    
    image = Image.open(filename).convert('RGB')
    image = np.array(image)
    if asTensor:
        image = imageToTensor(image)
        #image = image.unsqueeze(0)
    return image



def saveImage(image, filename, isTensor=False):
    '''
    Save an image to a file

    inputs
    image: image as PyTorch Tensor or NumPy array
    filename: path to save image file
    isTensor: if True, image is a PyTorch Tensor, else image is a NumPy array
    '''
    
    if isTensor:
        image = tensorToImage(image)
    image = Image.fromarray(image)
    image.save(filename)



def loadMask(filename, asTensor=False):
    '''
    Load a mask from a file

    inputs
    filename: path to mask file
    asTensor: if True, return mask as PyTorch Tensor, else return as NumPy array

    output
    mask: mask as PyTorch Tensor or NumPy array
    '''
    
    mask = Image.open(filename).convert('L')
    mask = np.array(mask)
    if asTensor:
        mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
    return mask



def saveMask(mask, filename, isTensor=False):
    '''
    Save a mask to a file

    inputs
    mask: mask as PyTorch Tensor or NumPy array
    filename: path to save mask file
    isTensor: if True, mask is a PyTorch Tensor, else mask is a NumPy array
    '''
    
    if isTensor:
        mask = tensorToMask(mask)
    mask = Image.fromarray(mask)
    mask.save(filename)



def imageToTensor(image,addBatch=True):
    '''
    Convert an image from a NumPy array to a PyTorch Tensor

    inputs
    image: image as NumPy array

    output
    tensor: image as PyTorch Tensor
    '''
    
    image = np.transpose(image, (2, 0, 1))  # Convert HxWxC to CxHxW
    if np.amax(image) > 1.0:
        image = image/255.0
    
    result = torch.from_numpy(image).float()
    if addBatch:
        result = result.unsqueeze(0)
    return result



def tensorToImage(tensor):
    '''
    Convert an image from a PyTorch Tensor to a NumPy array

    inputs
    tensor: image as PyTorch Tensor

    output
    image: image as NumPy array
    '''
    
    image = tensor.detach().cpu().numpy() * 255.0
    image = np.clip(image,0,255)
    image = image.squeeze(0)
    image = np.transpose(image, (1, 2, 0)).astype(np.uint8)  # Convert CxHxW to HxWxC
    return image



def toTensor(image):
    transform = transforms.ToTensor()
    return transform(image)



def toNumpy(tensor):
    if tensor.dim() == 4:  # Tensor with shape (B, C, H, W)
        tensor = tensor.squeeze(0)  # Remove the batch dimension if it exists
    if tensor.dim() == 3:  # Tensor with shape (C, H, W)
        return tensor.detach().cpu().numpy().transpose(1, 2, 0)
    else:
        raise ValueError("Unsupported tensor shape: {}".format(tensor.shape))



def maskToTensor(mask):
    '''
    Convert a mask from a NumPy array to a PyTorch Tensor

    inputs
    mask: mask as NumPy array

    output
    tensor: mask as PyTorch Tensor
    '''
    
    mask = torch.from_numpy(mask).unsqueeze(0).float()

    if torch.amax(mask) > 1.0:
        mask = mask/255.0

    return mask



def tensorToMask(tensor):
    '''
    Convert a mask from a PyTorch Tensor to a NumPy array

    inputs
    tensor: mask as PyTorch Tensor

    output
    mask: mask as NumPy array
    '''
    
    mask = tensor.detach().cpu().squeeze().numpy() * 255.0
    return mask.astype(np.uint8)



#def applyMask(image, mask):

    #alpha = np.mean(mask_im[:, :, :3], axis=2)  # Convert RGBA to grayscale
    #alpha[alpha > 0] = 1.0  # Set non-transparent regions to fully opaque
    #alpha = np.expand_dims(alpha, axis=2)  
    #return alpha

    #mask_temp = mask.squeeze()
    #mask_temp = cv2.resize(mask_temp.numpy(),(result.shape[-1],result.shape[-2]))
    #mask = torch.from_numpy(mask_temp).unsqueeze(0)
       
    #return image*mask



def applyMaskTensor(result, mask):

    # Check that mask is passed in
    if mask is None:
        return result

    # Check that mask and result are the same size
    if result.shape[-2] != mask.shape[-2] or result.shape[-1] != mask.shape[-1]:
        resizer = transforms.Resize((result.shape[-2],result.shape[-1]),transforms.InterpolationMode.NEAREST_EXACT,antialias=False)
        mask = resizer(mask)

    # Check that they have the same dimesions
    if mask.ndim < result.ndim:
        mask = mask.unsqueeze(-3)

    return result*mask



def alphaBlend(image, mask, background):
    '''
    Alpha blend an image with a mask

    inputs
    image: image as PyTorch Tensor or NumPy array
    mask: mask as PyTorch Tensor or NumPy array
    background: background image as PyTorch Tensor or NumPy array

    output
    blended: image alpha blended with background as PyTorch Tensor or NumPy array
    '''

    alpha = mask / mask.max()

    if isinstance(image, np.ndarray):
        alpha = cv2.resize(alpha,(image.shape[1],image.shape[0]))
        background = cv2.resize(background,(image.shape[1],image.shape[0]))

    # This makes sure the mask and image have the same number of dimensions
    if alpha.ndim < image.ndim:
        alpha = alpha[...,None]


    blended = image * alpha + background * (1 - alpha)
    return blended



def calcStats(im, mask=None, isTensor = True):

    # Check if tensor
    if isTensor:
        im = tensorToImage(im)
        if mask is not None:
            mask = tensorToMask(mask)

    # If no mask is provided, calculate stats for the entire image
    if mask is None:
        colors = im.reshape(-1, 3)  # Flatten the image into a 2D array of RGB values
    else:
        # Row and Column values of mask pixels
        locs = np.where(mask)
        #print(len(locs[0]))

        # RGB Values of image pixels in the mask
        colors = im[locs]
        #print(colors)

    # Normalize
    colors = colors/255
    print("Colors",colors)

    # Calculate standard deviation, and L1 norm for red channel
    std_red = np.std(colors[:,0])
    L1_red = np.mean(np.abs(colors[:,0] - np.mean(colors[:,0])))
    print(std_red,L1_red)

    # Convert to grayscale
    grayscale = 0.299*colors[:,0] + 0.587*colors[:,1] + 0.114*colors[:,2]

    # Calculate statistics for RGB and grayscale values:
    mean_red = np.mean(colors[:,0])
    mean_green = np.mean(colors[:, 1])
    mean_blue = np.mean(colors[:, 2])
    print(mean_red,mean_green,mean_blue)

    mean_gray = np.mean(grayscale)
    print(mean_gray)

    # Calculate variance for each channels
    var_red = np.mean((colors[:,0] - np.mean(colors[:,0]))**2)
    var_green = np.mean((colors[:,1] - np.mean(colors[:,1]))**2)
    var_blue = np.mean((colors[:,2] - np.mean(colors[:,2]))**2)
    print(var_red,var_green,var_blue)

    var_gray = np.mean((grayscale - np.mean(grayscale))**2)
    #variance = np.var(grayscale)
    print(var_gray)

    total_var_RGB = np.sqrt(var_red**2 + var_green**2 + var_blue**2)
    print(total_var_RGB)

    skewness_gray = np.mean((grayscale - np.mean(grayscale))**3)
    print(skewness_gray)

    kurtosis_gray = np.mean((grayscale - mean_gray) ** 4) / (np.var(grayscale) ** 2) - 3
    print(kurtosis_gray)

    # Calculate histograms for each channel
    hist_red = np.histogram(colors[:, 0], bins=256, range=(0, 1))[0]
    hist_green = np.histogram(colors[:, 1], bins=256, range=(0, 1))[0]
    hist_blue = np.histogram(colors[:, 2], bins=256, range=(0, 1))[0]
    print(hist_red,hist_green,hist_blue)

    hist_gray = np.histogram(grayscale, bins=256, range=(0, 1))[0]
    print(hist_gray)
    # hist_gray = np.histogram(grayscale,bins=8)
    # print(hist_gray)

    # Calculate brightness (mean of grayscale)
    brightness_gray = mean_gray
    print(brightness_gray)

    # Calculate contrast (difference between max and min of grayscale)
    contrast_gray = np.max(grayscale) - np.min(grayscale)
    print(contrast_gray)

    # # Calculate the entropy of the grayscale image
    # entropy_gray = exposure.shannon_entropy(grayscale)
    # print(entropy_gray)

    # Manual calculation of entropy
    value, counts = np.unique(grayscale, return_counts=True)
    probabilities = counts / len(grayscale)
    entropy_gray = -np.sum(probabilities * np.log2(probabilities))
    print(entropy_gray)

    #from styleloss import calculateStyleLoss
    #style_loss = calclulateStlyeLoss(...)

    # Save results to a dictionary
    filedata = {}
    filedata["num_pixels"] = len(colors)
    filedata["std_red"] = std_red
    filedata["L1_red"] = L1_red
    filedata["mean_red"] = mean_red
    filedata["mean_green"] = mean_green
    filedata["mean_blue"] = mean_blue
    filedata["mean_gray"] = mean_gray
    filedata["var_red"] = var_red
    filedata["var_green"] = var_green
    filedata["var_blue"] = var_blue
    filedata["var_gray"] = var_gray
    filedata["total_var_RGB"] = total_var_RGB
    filedata["skewness_gray"] = skewness_gray
    filedata["kurtosis_gray"] = kurtosis_gray
    filedata["hist_red"] = hist_red.tolist()
    filedata["hist_green"] = hist_green.tolist()
    filedata["hist_blue"] = hist_blue.tolist()
    filedata["hist_gray"] = hist_gray.tolist()
    filedata["brightness_gray"] = brightness_gray
    filedata["contrast_gray"] = contrast_gray
    filedata["entropy_gray"] = entropy_gray

    return filedata

    # filedata = {
    #     "num_pixels": len(colors),
    #     "mean_red": mean_red,
    #     "std_red": std_red,
    #     "L1_red": L1_red,
    #     "mean_gray": mean_gray,
    #     "var_red" : var_red,
    #     "var_green" : var_green,
    #     "var_blue" : var_blue,
    #     "var_gray" : var_gray,
    #     "total_var_RGB" : total_var_RGB,
    #     "skewness_gray" : skewness_gray,
    #     "kurtosis_gray" : kurtosis_gray,
    #     "hist_red" : hist_red[0].tolist(),
    #     "hist_green" : hist_green[0].tolist(),
    #     "hist_blue" : hist_blue[0].tolist(),
    #     "hist_gray" : hist_gray[0].tolist(),
    #     "brightness_gray" : brightness_gray,
    #     "contrast_gray" : contrast_gray,
    #     "entropy_gray" : entropy_gray

    # }

    

def saveStats(filedata, save_fn):
    import json
    import numpy as np

    # Function to convert non-serializable objects
    def convert(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()  # Convert numpy types to Python types
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

    # Serialize the filedata using the custom convert function
    json_object = json.dumps(filedata, indent=4, default=convert)

    # Writing to the specified JSON file
    with open(save_fn, "w") as outfile:
        outfile.write(json_object)

    # Print the JSON object
    print(json_object)



# def saveStats(filedata, save_fn):

#     # Save results to a JSON file
#     import json
#     json_object = json.dumps(filedata, indent=4)

#     # Writing to sample.json
#     with open(save_fn, "w") as outfile:
#         outfile.write(json_object)

#     # Print the results
#     print(json_object)



# class MyEncoder(nn.Module):
#     def __init__(self):
#         super(MyEncoder, self).__init__()
        
#         # Define the convolutional layers
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
#         # Additional layers can be defined similarly

#     def forward(self, x):
#         # Ensure 'x' is a Tensor (if necessary)
#         if isinstance(x, np.ndarray):
#             x = torch.from_numpy(x).float()
        
#         # Apply the layers defined in __init__
#         out = self.conv1(x)
#         out = F.relu(out)
#         out = self.conv2(out)
#         out = F.relu(out)
        
#         return out
