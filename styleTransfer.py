from libs.Matrix import MulLayer as MulLayer_reg
from libs.models import encoder4 as encoder4_reg, decoder4 as decoder4_reg
from libs.Matrix_masked import MulLayer as MulLayer_masked
from libs.models_masked import encoder4 as encoder4_masked, decoder4 as decoder4_masked


import torch
import utils

import numpy as np
import matplotlib.pyplot as plt

from pycocotools import mask as mask_utils

import json
import cv2


# Type 1: Linear Style Transfer (content, style)

def styletransfer(content, style):
    '''
    Original style transfer function

    inputs
    content: content image as PyTorch Tensor
    style: style image as PyTorch Tensor

    output
    styled: styled image as PyTorch Tensor
    '''

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    enc_ref = encoder4_reg()
    dec_ref = decoder4_reg()
    matrix_ref = MulLayer_reg('r41')

    enc_ref.load_state_dict(torch.load('models/vgg_r41.pth'))
    dec_ref.load_state_dict(torch.load('models/dec_r41.pth'))
    matrix_ref.load_state_dict(torch.load('models/r41.pth',map_location=torch.device('cpu')))


    with torch.no_grad():
        # Reference comparison
        cF_ref = enc_ref(content)
        sF_ref = enc_ref(style)
        feature_ref,transmatrix_ref = matrix_ref(cF_ref['r41'],sF_ref['r41'])
        result = dec_ref(feature_ref)
        

    return result


# Type 2: Linear Style Transfer (content, style, mask) (Regular) Mask_Window

def style_then_mask(content, style, mask=None):
    '''
    Original style transfer function, followed by masking operation

    inputs
    content: content image as PyTorch Tensor
    style: style image as PyTorch Tensor
    mask: mask image as PyTorch Tensor, if None is provided, return the standard styled image

    output
    styled: styled image alpha blended with original background as PyTorch Tensor
    '''
    result = styletransfer(content,style)

    return utils.applyMaskTensor(result,mask)


# Type 3: Linear Style Transfer (content, mask, style) (Regular) Mask_Window

# def mask_then_style(content, style, mask=None):
#     content = utils.applyMask(content,mask)
#     return styletransfer(content,style)


def mask_then_style(content, style, mask=None):
    '''
    Masking operation, followed by style transfer function

    inputs
    content: content image as PyTorch Tensor
    style: style image as PyTorch Tensor
    mask: mask image as PyTorch Tensor, if None is provided, return the standard styled image

    output
    styled: styled image alpha blended with original background as PyTorch Tensor
    '''
    
    content = utils.applyMaskTensor(content,mask)
    
    return styletransfer(content,style)

# Type 4: Linear Style Transfer (content, style, mask)(Partial convolution) Mask_bird (4 types included)

def style_with_partialConv(content, style, mask=None):
    '''
    Masked style transfer function using Partial Convolution
    Make sure to use the masked library files

    inputs
    content: content image as PyTorch Tensor
    style: style image as PyTorch Tensor
    mask: mask image as PyTorch Tensor, if None is provided, return the standard styled image

    output
    styled: styled image alpha blended with original background as PyTorch Tensor
    '''
    
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    enc_ref = encoder4_masked()
    dec_ref = decoder4_masked()
    matrix_ref = MulLayer_masked('r41')

    enc_ref.load_state_dict(torch.load('models/vgg_r41.pth'))
    dec_ref.load_state_dict(torch.load('models/dec_r41.pth'))
    matrix_ref.load_state_dict(torch.load('models/r41.pth', map_location=torch.device('cpu')))

    with torch.no_grad():
        # Reference comparison
        cF_ref,small_mask = enc_ref(content,mask)
        sF_ref,_ = enc_ref(style)
        feature_ref,transmatrix_ref = matrix_ref(cF_ref['r41'],sF_ref['r41'],small_mask)
        result = dec_ref(feature_ref,mask)

    return result