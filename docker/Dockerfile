FROM docker.io/ultralytics/ultralytics:latest

WORKDIR /ultralytics_ws

# ユーザー指定
ARG UID
ARG GID
ARG USERNAME
ARG GROUPNAME

RUN groupadd -g $GID $GROUPNAME
RUN useradd -m -u $UID -g $GID $USERNAME

RUN chown $USERNAME:$GROUPNAME -R /ultralytics_ws
RUN chown $USERNAME:$GROUPNAME -R /ultralytics

USER $USERNAME

# ターミナルの色指定
RUN echo "PS1='\[\e[41;37m\]\u@\h\[\e[0m\]:\[\e[34m\]\w\[\e[0m\]$ '" >> /home/$USERNAME/.bashrc

ENTRYPOINT ["/bin/bash"]