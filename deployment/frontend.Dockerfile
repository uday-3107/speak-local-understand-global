FROM nginx:1.27-alpine
COPY index.html /usr/share/nginx/html/index.html
COPY app.html /usr/share/nginx/html/app.html
COPY dashboard.html /usr/share/nginx/html/dashboard.html
COPY lecture.html /usr/share/nginx/html/lecture.html
COPY css/ /usr/share/nginx/html/css/
COPY js/ /usr/share/nginx/html/js/
COPY deployment/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80