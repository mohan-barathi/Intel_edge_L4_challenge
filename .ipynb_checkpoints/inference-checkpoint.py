import argparse
import cv2
import numpy as np
import time
from helpers import load_to_IE, preprocessing
import glob

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Load an IR into the Inference Engine")
    # -- Create the descriptions for the commands
    m_desc = "The location of the model XML file"
    i_desc = "The location of the image input"
    r_desc = "The type of inference request: Async ('A') or Sync ('S')"

    # -- Create the arguments
    parser.add_argument("-m", help=m_desc)
    parser.add_argument("-i", help=i_desc)
    parser.add_argument("-r", help=i_desc)
    args = parser.parse_args()

    return args


def async_inference(exec_net, input_blob, image):
    ### TODO: Add code to perform asynchronous inference
    ### Note: Return the exec_net
    exec_net.start_async(request_id=0, inputs={input_blob: image})
    while True:
        status = exec_net.requests[0].wait(-1)
        if status == 0:
            break
        else:
            time.sleep(1)
    return exec_net


def sync_inference(exec_net, input_blob, image):
    ### TODO: Add code to perform synchronous inference
    ### Note: Return the result of inference
    result = exec_net.infer({input_blob: image})

    return result

def draw_points(image, points, draw=True):
    white = [255,255,255]
    #image[10,5]=red
    
    for x, y in zip(*[iter(points)]*2):
        image[48*x, 48*y] = white
        
    return image

def perform_inference(exec_net, request_type, input_image, input_shape):
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

    # Perform either synchronous or asynchronous inference
    request_type = request_type.lower()
    if request_type == 'a':
        output = async_inference(exec_net, input_blob, preprocessed_image)
    elif request_type == 's':
        output = sync_inference(exec_net, input_blob, preprocessed_image)
        img_output = draw_points(preprocessed_image, output)
    else:
        print("Unknown inference request type, should be 'A' or 'S'.")
        exit(1)

    # Return the output for testing purposes
    return img_output


def main():
    args = get_args()
    print (args)
    input_images = glob.glob("input_images/*.jpg")
    print (input_images)
    exec_net, input_shape = load_to_IE(args.m, CPU_EXTENSION)
    
    for image in input_images:
        result = perform_inference(exec_net, 's', image, input_shape)
        result = result.reshape(3, height, width)
        print (result.shape)
        cv2.imwrite("outputs/{}-output.png".format(image.split("/")[1]), result)

if __name__ == "__main__":
    main()
