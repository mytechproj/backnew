from django.shortcuts import render,redirect
from django.http import HttpResponse
from backgroundapp.models import Contact
from datetime import datetime
from django.contrib import messages
from django.contrib.auth import authenticate
from django.contrib.auth import logout,login
from django.contrib.auth.models import User
from django.core.files.storage import FileSystemStorage
from .pytemplates.mymodel import main
from .pytemplates.mymodel import fnfilepath
from mywebproject import settings
from django.urls import reverse
import secrets 
import string 
import cv2
from PIL import Image
  
# initializing size of string  
N = 10

# Create your views here.
def home_view(request):
    #return HttpResponse("<h1> Welcome to Django!!!!</h1>")
    if request.user.is_anonymous:
        return render(request,"login.html")
    contex = {"variable":"verygood"}
    return render(request,"index.html",contex)
def about(request):
    #return render(request,"services.html")
    contex = {}
    if request.method =="POST":     
        myfile = request.FILES.get("file")        
        fs = FileSystemStorage(location=settings.MYMEDIA_ROOT)
        res = ''.join(secrets.choice(string.ascii_lowercase + string.digits) 
                                                  for i in range(N))
        name=fs.save(res+myfile.name,myfile)
        #name=fs.save("dog1.jpg",myfile)
        print ("name="+name)
        #print ("url="+fs.url(name))
        #context['url'] = fs.url(name)
        main(name)
        #origfile=main()    
        #imgurl=fnfilepath()
        origfile=name
        aaa = origfile.split(".")
        remfile=aaa[0]+"-rem.png"        
        request.session['origfile'] = origfile
        request.session['remfile'] = remfile
        origfile = request.session.get('origfile', '')
        remfile = request.session.get('remfile', '')   
        #-------------------------------------- opencv        
        #background = cv2.imread("background.png", cv2.IMREAD_UNCHANGED)
        #foreground = cv2.imread("/mymedia/"+remfile, cv2.IMREAD_UNCHANGED)
        #background = cv2.imread("/mymedia/backg.png", cv2.IMREAD_UNCHANGED)
        
        """
        background = cv2.imread("E:/myproject/mywebproject/backgroundapp/pytemplates/images/backg.png")
        overlay = cv2.imread("E:/myproject/mywebproject/backgroundapp/pytemplates/images/"+remfile)
        print("background=")
        print(background.shape)
        print("overlay=")
        print(overlay.shape)

        added_image = cv2.addWeighted(background,0.4,overlay,0.1,0)
        """
        """
        # display the image
        #cv2.imshow("Composited image", background)
        #---------------------------------------------
        #background = cv2.imread("E:/myproject/mywebproject/backgroundapp/pytemplates/images/backg.jpg", cv2.IMREAD_UNCHANGED)
        background = "E:/myproject/mywebproject/backgroundapp/pytemplates/images/backg.jpg"
        foreground = cv2.imread("E:/myproject/mywebproject/backgroundapp/pytemplates/images/"+remfile, cv2.IMREAD_UNCHANGED)
        a = background.split(".")
        if(a[1]=="jpg"):
            background1=cv2.imread(background, 1)
            cv2.imwrite("E:/myproject/mywebproject/backgroundapp/pytemplates/images/backg.png",background1 )

            #im = Image.open(background)
            #im.save("E:/myproject/mywebproject/backgroundapp/pytemplates/images/backg.png")      
       
        #cv2.imwrite("E:/myproject/mywebproject/backgroundapp/pytemplates/images/backg.png", cv2.imread(background, 1))
        background = cv2.imread("E:/myproject/mywebproject/backgroundapp/pytemplates/images/backg.png", cv2.IMREAD_UNCHANGED)
        background = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)
        print("background=")
        print(background.shape)
        print("foreground=")
        print(foreground.shape)
        height, width, channels = foreground.shape
        background = cv2.resize(background, (width, height)) 
        # normalize alpha channels from 0-255 to 0-1
        alpha_background = background[:,:,3] / 255.0
        alpha_foreground = foreground[:,:,3] / 255.0

        # set adjusted colors
        for color in range(0, 3):
            background[:,:,color] = alpha_foreground * foreground[:,:,color] + \
                alpha_background * background[:,:,color] * (1 - alpha_foreground)

        # set adjusted alpha and denormalize back to 0-255
        background[:,:,3] = (1 - (1 - alpha_foreground) * (1 - alpha_background)) * 255

        cv2.imwrite("E:/myproject/mywebproject/backgroundapp/pytemplates/images/finalimg.png", background)
        image1 = cv2.imread('E:/myproject/mywebproject/backgroundapp/pytemplates/images/finalimg.png')

        # Save .jpg image
        cv2.imwrite('E:/myproject/mywebproject/backgroundapp/pytemplates/images/finalimg-good.jpg', image1, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        cv2.waitKey(0)

        #--------------------------------------
        """




        contex = {"origfile":"/mymedia/"+origfile,"remfile":"/mymedia/"+remfile,"origfilename":origfile,}       
        return render(request,"about.html",contex)
        
    origfile = request.session.get('origfile', '')
    remfile = request.session.get('remfile', '')   
    contex = {"origfile":"/mymedia/"+origfile,"remfile":"/mymedia/"+remfile,"origfilename":origfile,}
    return render(request,"about.html",contex)    
def services(request): 
    
    return render(request,"services.html")
def loginuser(request):
    if request.method == "POST":
        #username pwd
        #sushil online200
        email = request.POST.get("email")
        pwd = request.POST.get("password")
        user = authenticate(username=email, password=pwd)
        if user is not None:
            login(request,user)
            return redirect("/")
        else:
            return render(request,"login.html")
    return redirect("/")

    
def logoutuser(request):
        logout(request)
        return redirect("/login")
def contact(request):
    if request.method == "POST":
        name = request.POST.get("name")
        email = request.POST.get("email")
        desc = request.POST.get("desc")
        contact = Contact(name=name,email=email,desc=desc,date=datetime.today())
        contact.save()
        messages.success(request, 'Your Message has been sent !!')

    return render(request,"contact.html")
    