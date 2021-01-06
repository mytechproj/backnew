"""
WSGI config for mywebproject project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.1/howto/deployment/wsgi/
"""

import os, sys
#add project path
#sys.path.append('E:\myproject\mywebproject')
# add the virtualenv site-packages path to the sys.path
#sys.path.append('E:\myproject\myprojvenv\Lib\site-packages')



from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mywebproject.settings')

application = get_wsgi_application()
