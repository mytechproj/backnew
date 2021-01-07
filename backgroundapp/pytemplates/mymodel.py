import os

def main(imgname):

    # --------- 1. get image path and name ---------
	
    model_name='u2netp'# change to u2netp	
    #imgname = "dog1.jpg"
    print("here0000") 
    image_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pytemplates','images',imgname) # changed to 'images' directory which is populated while running the script
    #image_dir = "E:\\DjangoDeepLearning\\djangotutorial\\static\\img" # changed to 'images' directory which is populated while running the script
    prediction_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pytemplates','results')+ os.sep # changed to 'results' directory which is populated after the predictions
    #prediction_dir = "E:\\DjangoDeepLearning\\djangotutorial\\static\\img" # changed to 'results' directory which is populated after the predictions
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
	'pytemplates','u2netp.pth') # path to u2netp pretrained weights

    #img_name_list = glob.glob(image_dir + os.sep + '*')      
    img_name_list = glob.glob(image_dir)   
    print("here1")    
    #return 'test.png'
        
   
    
    
    
