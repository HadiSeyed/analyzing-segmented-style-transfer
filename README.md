# analyzing-segmented-style-transfer

Repo for Analyzing Style Transfer Algorithms for Segmented Images.

## analyzing-segmented-style-transfer

I work with [Dr. David Hart](http://davidhartcv.com), assistant professor of computer science at East Carolina University, in the field of Computer Vision research. My thesis topic investigates different approaches to style transfer and integrating the Segment Anything Model with style transfer. Specifically, it proposes Partial Convolution as a way to improve style transfer for segmented regions. Additionally, how different style transfer techniques are affected by different mask sizes, image statistics, etc. Example outputs from my research are shown below.  

### Figure 1: Example output using the Segment Anything Model. Given an input image (left) and a point from the user, the model can determine all pixels associated with the object closest surrounding that point, providing a mask of the object (right).
![figure 1](<1. Segment Anything Model.png>)

### Figure 2: Example output using the Linear Style Transfer algorithm for a full image.  A content and style image (left) are fed into the style transfer network, resulting in stylized version of the content image (right).
![figure 2](<2. Linear Style Transfer.jpg>)

### Figure 3: Example output using a Partial Convolution algorithm for style transfer. A content, style image (left) and a given masked object (the bird), the partial convolution style transfer algorithm can apply the stylization to the masked region exclusively (right).
![figure 3](<3. Partial Convolution + SAM.jpg>)

### Figure 4: Example output using the Linear Style Transfer algorithm for a full image, followed by masking (Style-then-Mask). A content image and style image (left) give the following stylized output for the masked region (right). The result stylization tends to be darker than the input style features.
![figure 4](<4. Style-then-Mask.jpg>)

### Figure 5: Example output using the Linear Style Transfer algorithm for a full image, followed by masking (Mask-then-Style). A content image and style image (left) give the following stylized output for the masked region (right). The result stylization tends to be much brighter than the input style features.
![figure 5](<5. Mask-then-Style.jpg>)

### Figure 6: Example output using the Partial Convolution algorithm. A content image and style image (left) give the following stylized output for the masked region (right). The result stylization tends to be closer to the input style features than the other two approaches.
![figure 6](<6. Partial Convolution.png>)

### Figure 7: Comparison of the three techniques: style-then-mask (left), mask-then-style (middle), partial convolution (right). A content image and style image (top) give the following stylized output for the masked region (bottom).
![figure 7](<7. Comparison.png>)
