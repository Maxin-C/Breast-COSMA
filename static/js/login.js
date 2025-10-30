// static/js/login.js
document.addEventListener('DOMContentLoaded', function() {
    const loginForm = document.getElementById('login-form');
    const errorMessage = document.getElementById('error-message');

    loginForm.addEventListener('submit', function(event) {
        event.preventDefault();
        errorMessage.textContent = ''; // 清空之前的错误信息

        const username = document.getElementById('username').value;
        const password = document.getElementById('password').value;

        fetch('/api/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ username, password })
        })
        .then(response => {
            if (response.ok) {
                // 登录成功，跳转到主页
                window.location.href = '/'; 
            } else {
                return response.json(); // 获取错误信息
            }
        })
        .then(data => {
            if (data && data.error) {
                errorMessage.textContent = data.error;
            }
        })
        .catch(error => {
            console.error('Login error:', error);
            errorMessage.textContent = '登录时发生网络错误，请稍后重试。';
        });
    });
});