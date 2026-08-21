FROM nginx:1.27-alpine
COPY frontend/index.html /usr/share/nginx/html/index.html
COPY frontend/app.html /usr/share/nginx/html/app.html
COPY frontend/dashboard.html /usr/share/nginx/html/dashboard.html
COPY frontend/lecture.html /usr/share/nginx/html/lecture.html
COPY frontend/css/ /usr/share/nginx/html/css/
COPY frontend/js/ /usr/share/nginx/html/js/
COPY deployment/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80