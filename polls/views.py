from django.http import HttpResponse
from django.shortcuts import render
from django.http.response import StreamingHttpResponse
from polls.camera import VideoCamera, gen

cam = VideoCamera()

def index(request):
    return render(request, 'index.html')

def video_feed(request):
	return StreamingHttpResponse(gen(cam), content_type='multipart/x-mixed-replace; boundary=frame')
