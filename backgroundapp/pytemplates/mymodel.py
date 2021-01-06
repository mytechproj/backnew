import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob
#import cv2

from .data_loader import RescaleT
from .data_loader import ToTensor
from .data_loader import ToTensorLab
from .data_loader import SalObjDataset

from .model import U2NET # full size version 173.6 MB
from .model import U2NETP # small version u2net 4.7 MB

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]    
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir+imidx+'.png')
def fnfilepath(strpath):
    return strpath
def main(imgname):

    # --------- 1. get image path and name ---------
    model_name='u2netp'# change to u2netp
    #imgname = "dog1.jpg"
    image_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pytemplates','images',imgname) # changed to 'images' directory which is populated while running the script
    #image_dir = "E:\\DjangoDeepLearning\\djangotutorial\\static\\img" # changed to 'images' directory which is populated while running the script
    prediction_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pytemplates','results')+ os.sep # changed to 'results' directory which is populated after the predictions
    #prediction_dir = "E:\\DjangoDeepLearning\\djangotutorial\\static\\img" # changed to 'results' directory which is populated after the predictions
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
	'pytemplates','u2netp.pth') # path to u2netp pretrained weights

    #img_name_list = glob.glob(image_dir + os.sep + '*')      
    img_name_list = glob.glob(image_dir)   
    
    # --------- 2. dataloader ---------
    #1. dataloader
    
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )    
        
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=0)
    
    # --------- 3. model define ---------
    net = U2NETP(3,1)
    
    if torch.cuda.is_available():        
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
        net.eval()
    else:        
        net.load_state_dict(torch.load(model_dir,map_location="cpu"))
        net.eval()        
    
    # --------- 4. inference for each image ---------    
    for i_test, data_test in enumerate(test_salobj_dataloader):                
        print("inferencing:",img_name_list[i_test].split(os.sep)[-1])        

        inputs_test = data_test['image']        
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)
        
        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        save_output(img_name_list[i_test],pred,prediction_dir)

        del d1,d2,d3,d4,d5,d6,d7
    #------ process files
    #image_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pytemplates'+ os.sep +'images') # changed to 'images' directory which is populated while running the script    
     #names = [name[:-4] for name in os.listdir(image_dir)]
    names = [name[:-4] for name in img_name_list] #uma   
    THRESHOLD = 0.9
    RESCALE = 255
    LAYER = 2
    COLOR = (0, 0, 0)
    THICKNESS = 4
    SAL_SHIFT = 100

    for name in names:
        name = name[name.rindex(os.sep)+1:] #uma for splitting name        
        # BACKGROUND REMOVAL

        #if name == '.ipynb_checkpo':
        #    continue
        #image_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pytemplates'+ os.sep +'images') # changed to 'images' directory which is populated while running the script
        
        output = load_img(prediction_dir+name+'.png')
        #output = load_img('/content/newbackground/results/'+name+'.png')
        out_img = img_to_array(output)
        out_img /= RESCALE
        out_img[out_img > THRESHOLD] = 1
        out_img[out_img <= THRESHOLD] = 0

        shape = out_img.shape
        a_layer_init = np.ones(shape = (shape[0],shape[1],1))
        mul_layer = np.expand_dims(out_img[:,:,0],axis=2)
        a_layer = mul_layer*a_layer_init
        rgba_out = np.append(out_img,a_layer,axis=2)

        #input = load_img(image_dir+ os.sep +name+'.jpg') uma
        #input = load_img(image_dir+ os.sep +imgname)
        input = load_img(image_dir)
        inp_img = img_to_array(input)
        inp_img /= RESCALE

        a_layer = np.ones(shape = (shape[0],shape[1],1))
        rgba_inp = np.append(inp_img,a_layer,axis=2)

        rem_back = (rgba_inp*rgba_out)
        rem_back_scaled = rem_back*RESCALE
        """
        # BOUNDING BOX CREATION

        out_layer = out_img[:,:,LAYER]
        x_starts = [np.where(out_layer[i]==1)[0][0] if len(np.where(out_layer[i]==1)[0])!=0 else out_layer.shape[0]+1 for i in range(out_layer.shape[0])]
        x_ends = [np.where(out_layer[i]==1)[0][-1] if len(np.where(out_layer[i]==1)[0])!=0 else 0 for i in range(out_layer.shape[0])]
        y_starts = [np.where(out_layer.T[i]==1)[0][0] if len(np.where(out_layer.T[i]==1)[0])!=0 else out_layer.T.shape[0]+1 for i in range(out_layer.T.shape[0])]
        y_ends = [np.where(out_layer.T[i]==1)[0][-1] if len(np.where(out_layer.T[i]==1)[0])!=0 else 0 for i in range(out_layer.T.shape[0])]
        
        startx = min(x_starts)
        endx = max(x_ends)
        starty = min(y_starts)
        endy = max(y_ends)
        start = (startx,starty)
        end = (endx,endy)

        box_img = inp_img.copy()
        box_img = cv2.rectangle(box_img, start, end, COLOR, THICKNESS)
        box_img = np.append(box_img,a_layer,axis=2)
        box_img_scaled = box_img*RESCALE

        # SALIENT FEATURE MAP

        sal_img = inp_img.copy()
        add_layer = out_img.copy()
        add_layer[add_layer==1] = SAL_SHIFT/RESCALE
        sal_img[:,:,LAYER] += add_layer[:,:,LAYER]
        sal_img = np.append(sal_img,a_layer,axis=2)
        sal_img_scaled = sal_img*RESCALE
        sal_img_scaled[sal_img_scaled>RESCALE] = RESCALE
        """
        # OUTPUT RESULTS

        inp_img*=RESCALE
        inp_img = np.append(inp_img,RESCALE*a_layer,axis=2)
        #inp_img = cv2.resize(inp_img,(int(shape[1]/3),int(shape[0]/3)))
        rem_back = rem_back_scaled
        result_img = Image.fromarray(rem_back.astype('uint8'), 'RGBA')
        #final_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pytemplates'+ os.sep +'finalresults'+ os.sep)
        final_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pytemplates'+ os.sep +'images'+ os.sep)
        result_img.save(final_dir+name+'-rem.png')
        return name+'.png'
        #fnfilepath(final_dir+name+'.png')

        #display(result_img)
   

    
    
    