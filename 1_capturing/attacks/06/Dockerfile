FROM node:lts

# attack 6
RUN npm i -g autocannon

# attack 10
RUN curl https://k8s.adriansoftware.de/attacks/10/kuma-attack.js -o kuma-attack.js

RUN npm i -g socket.io-client

ENTRYPOINT  ["/bin/bash"]
CMD ["-c", "while :; do sleep 1; done"]
