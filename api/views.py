from django.shortcuts import render
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from BusDetector.obj_tracking import bus_search

def hello(request):
    return JsonResponse({'message': 'Hello from Django!'})

@csrf_exempt
def bus(request):
    response = {
        'status': 'ok',
        'message': 'This is a POST request',
    }
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            print(data)
            bus_search(data)
        except json.JSONDecodeError:
            response = {
                'status': 'error',
                'message': 'Invalid JSON data'
            }
    return JsonResponse(response)