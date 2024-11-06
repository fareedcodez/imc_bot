from django.shortcuts import render
from django.http import JsonResponse
from .training.bot_model import Chatbot, ChatbotTrainer
import os
from django.conf import settings
import json

# Define paths
MODEL_PATH = os.path.join(settings.BASE_DIR, 'chatbot_app', 'training', 'model.h5')
TRAINING_DATA_PATH = os.path.join(settings.BASE_DIR, 'chatbot_app', 'training', 'training_data.json')

def train_chatbot():
    """Function to train the chatbot"""
    trainer = ChatbotTrainer(TRAINING_DATA_PATH)
    trainer.train(MODEL_PATH)

def chatbot_view(request):
    # Train the model if it doesn't exist
    if not os.path.exists(MODEL_PATH):
        train_chatbot()
    
    if request.method == 'POST':
        user_input = request.POST.get('user_input')
        
        try:
            # Initialize chatbot
            chatbot = Chatbot(MODEL_PATH, TRAINING_DATA_PATH)
            
            # Get response from chatbot
            response = chatbot.get_response(user_input)
            
            # If it's an AJAX request, return JSON response
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'response': response})
            
            # For regular POST requests, return full page
            context = {
                'messages': [
                    {'content': user_input, 'is_user': True},
                    {'content': response, 'is_user': False}
                ]
            }
            return render(request, 'chatbot_app/chatbot.html', context)
            
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'error': error_message}, status=500)
            return render(request, 'imc_bot/index.html', {'error': error_message})
    
    # GET request - show empty chat
    return render(request, 'imc_bot/index.html')