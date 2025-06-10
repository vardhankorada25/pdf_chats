css='''
<style>

.chat-message {
    padding: 1.2rem 1.5rem;
    border-radius: 1rem;
    margin-bottom: 1rem;
    display: flex;
    max-width: 85%;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    font-size: 1rem;
    line-height: 1.6;
}


.chat-message.user {
    background: linear-gradient(135deg, #b2ebf2, #e0f7fa);
    align-self: flex-end;
    margin-left: auto;
    color: #004d40;
    justify-content: flex-end;
    text-align: right;
    border: 1px solid #b2ebf2;
}


.chat-message.bot {
    background: linear-gradient(135deg, #f1f8e9, #dcedc8);
    align-self: flex-start;
    margin-right: auto;
    color: #33691e;
    justify-content: flex-start;
    text-align: left;
    border: 1px solid #dcedc8;
}


.chat-message .message-content {
    max-width: 100%;
    word-wrap: break-word;
}


.chat-message .message-content p {
    margin: 0;
    white-space: pre-wrap;
}


@keyframes fadeIn {
    from {opacity: 0; transform: translateY(10px);}
    to {opacity: 1; transform: translateY(0);}
}
.chat-message {
    animation: fadeIn 0.4s ease-in-out;
}
</style>
  
'''
bot_template='''
<div class="chat-message bot">
    <div class="message-content">
        <p>{{MSG}}</p>
    </div>
</div>

'''
user_template='''
<div class="chat-message user">
    <div class="message-content">
        <p>{{MSG}}</p>
    </div>
</div>

'''