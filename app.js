document.addEventListener('DOMContentLoaded', () => {
    const chatIcon = document.getElementById('chat-icon');
    const chatWindow = document.getElementById('chat-window');
    const sendButton = document.getElementById('send-button');
    const userInput = document.getElementById('user-input');
    const chatMessages = document.getElementById('chat-messages');

    chatIcon.addEventListener('click', () => {
        chatWindow.classList.toggle('show');
        userInput.focus();
    });

    sendButton.addEventListener('click', () => {
        const message = userInput.value.trim();
        if (message) {
            appendMessage('user', message);
            userInput.value = '';
            fetchChatbotResponse(message);
        }
    });

    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendButton.click();
        }
    });

    function fetchChatbotResponse(message) {
        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message }),
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not successful');
            }
            return response.json();
        })
        .then(data => {
            appendMessage('bot', data.response);
        })
        .catch(error => {
            console.error('Error fetching response:', error);
            appendMessage('bot', "Sorry, I couldn't understand that.");
        });
    }

    function appendMessage(sender, message) {
        const messageElement = document.createElement('div');
        messageElement.className = `${sender}-message`;
        messageElement.textContent = message;
        chatMessages.appendChild(messageElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
});