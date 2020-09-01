FROM python:3.8

# Setup environment
RUN  cp /usr/local/bin/pip3.8 /usr/local/bin/pip3  # reenable pip3
RUN pip3 install --upgrade pip
WORKDIR /usr/src/app

# Install requirements
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

VOLUME ["/usr/src/app"]

CMD ["/usr/src/app/entry_point.sh"]
