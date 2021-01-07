from django.shortcuts import render,redirect
from django.http import HttpResponse
from backgroundapp.models import Contact
from datetime import datetime
from django.contrib import messages
from django.contrib.auth import authenticate
from django.contrib.auth import logout,login
from django.contrib.auth.models import User
from django.core.files.storage import FileSystemStorage
#from .pytemplates.mymodel import main
#from .pytemplates.mymodel import fnfilepath
from mywebproject import settings
from django.urls import reverse
import secrets 
import string 
#import cv2
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
    contex = {}
    if request.method =="POST":     
        myfile = request.FILES.get("file")        
        fs = FileSystemStorage(location=settings.MYMEDIA_ROOT)
        res = ''.join(secrets.choice(string.ascii_lowercase + string.digits) 
                                                  for i in range(N))
        name=fs.save(res+myfile.name,myfile)
        #name=fs.save("dog1.jpg",myfile)
        print ("nameimp="+name)
        #print ("url="+fs.url(name))
        #context['url'] = fs.url(name)
        main(name)
        print ("coming here")
        #origfile=main()    
        #imgurl=fnfilepath()
        origfile=name
        aaa = origfile.split(".")
        remfile=aaa[0]+"-rem.png"        
        request.session['origfile'] = origfile
        request.session['remfile'] = remfile
        origfile = request.session.get('origfile', '')
        remfile = request.session.get('remfile', '')       
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
    
