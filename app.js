document.getElementById('chat-icon').addEventListener('click', function() {
    const chatWindow = document.getElementById('chat-window');
    chatWindow.classList.toggle('visible');
});

document.getElementById('send-button').addEventListener('click', async function() {
    const userInput = document.getElementById('user-input').value.trim();
    if (!userInput) return;

    const chatMessages = document.getElementById('chat-messages');
    chatMessages.innerHTML += `<div class="user-message">${userInput}</div>`;

    document.getElementById('user-input').value = '';

    try {
        const response = await fetch('/chatbot', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: userInput })
        });

        if (!response.ok) {
            throw new Error("Network response was not ok.");
        }

        const data = await response.json();
        chatMessages.innerHTML += `<div class="bot-message">${data.response}</div>`;

        chatMessages.scrollTop = chatMessages.scrollHeight;
    } catch (error) {
        console.error('Error:', error);
        chatMessages.innerHTML += `<div class="bot-message">Error: Unable to connect to the chatbot.</div>`;
    }
});

document.getElementById('user-input').addEventListener('keydown', function(event) {
    if (event.key === 'Enter') {
        document.getElementById('send-button').click();
    }
});