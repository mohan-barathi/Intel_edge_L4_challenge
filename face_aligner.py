import cv2
import numpy as np
import time
from helpers import load_to_IE, preprocessing
import glob
import argparse

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Load an IR into the Inference Engine")
    # -- Create the descriptions for the commands
    m_desc = "The location of the model XML file"

    # -- Create the arguments
    parser.add_argument("-m", help=m_desc)
    args = parser.parse_args()
    return args

def sync_inference(exec_net, input_blob, image):
    result = exec_net.infer({input_blob: image})
    return result

def draw_points(image, points, draw=True):
    white = [255,255,255]
    for x, y in zip(*[iter(points)]*2):
        image[48*x, 48*y] = white
        
    return image

def perform_inference(exec_net, input_image, input_shape):
    '''
    Performs inference on an input image, given an ExecutableNetwork
    '''
    # Get input image
    image = cv2.imread(input_image)
    print ("raw image:", image.shape)
    # Extract the input shape
    n, c, h, w = input_shape
    # Preprocess it (applies for the IRs from the Pre-Trained Models lesson)
    preprocessed_image = preprocessing(image, h, w)

    # Get the input blob for the inference request
    input_blob = next(iter(exec_net.inputs))

    # Perform either synchronous
    output = sync_inference(exec_net, input_blob, preprocessed_image)
    img_output = draw_points(preprocessed_image, output)
    img_output = img_output.reshape(3, 48, 48)
    img_output = img_output.transpose((1,2,0))
    img_output = cv2.resize(img_output, (image.shape[1], image.shape[0]))


    # Return the output for testing purposes
    return img_output


def main():
    args = get_args()
    input_images = glob.glob("input_images/*.jpg")
    print (input_images)
    exec_net, input_shape = load_to_IE(args.m, CPU_EXTENSION)
    
    for image in input_images:
        result = perform_inference(exec_net, image, input_shape)
        print (result.shape)
        cv2.imwrite("output_images/{}-output.png".format(image.split("/")[1]), result)

if __name__ == "__main__":
    main()
