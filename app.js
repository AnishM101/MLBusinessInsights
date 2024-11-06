document.addEventListener("DOMContentLoaded", function() {
    console.log("JavaScript loaded");
    const chatIcon = document.getElementById('chat-icon');
    
    if (chatIcon) {
        chatIcon.addEventListener('click', function() {
            const chatWindow = document.getElementById('chat-window');
            chatWindow.classList.toggle('visible');
            console.log("Chat icon clicked");
        });
    }
    else {
        console.error("Chat icon not found");
    }

    const sendButton = document.getElementById('send-button');
    const userInput = document.getElementById('user-input');
    const chatMessages = document.getElementById('chat-messages');

    if (sendButton && userInput && chatMessages) {
        sendButton.addEventListener('click', async function() {
            const userMessage = userInput.value.trim();
            if (!userMessage) return;
            chatMessages.innerHTML += `<div class="user-message">${userMessage}</div>`;
            userInput.value = '';

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: userMessage })
                });

                if (!response.ok) {
                    throw new Error("Network response was not ok.");
                }

                const data = await response.json();
                chatMessages.innerHTML += `<div class="bot-message">${data.response}</div>`;
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
            catch (error) {
                console.error('Error:', error);
                chatMessages.innerHTML += `<div class="bot-message">Error: Unable to connect to the chatbot.</div>`;
            }
        });

        userInput.addEventListener('keydown', function(event) {
            if (event.key === 'Enter') {
                sendButton.click();
            }
        });
    }
    else {
        console.error("Chat components not found");
    }
});