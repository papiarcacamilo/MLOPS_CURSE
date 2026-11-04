# =============================================================================
# MLOPS_CURSE | Jenkins local para el pipeline de aprobacion
# =============================================================================
#
# Por que una imagen propia y no la oficial tal cual: la guia instala el cliente
# de Docker a mano dentro del contenedor, y eso se pierde cada vez que se recrea.
# Aqui queda declarado, igual que los plugins.
# =============================================================================

FROM jenkins/jenkins:2.580.2-lts-jdk21

USER root

# Solo el cliente: el motor es el de Docker Desktop, al que se llega por el socket.
RUN apt-get update \
    && apt-get install -y --no-install-recommends docker-cli \
    && rm -rf /var/lib/apt/lists/*

COPY plugins.txt /usr/share/jenkins/ref/plugins.txt
RUN jenkins-plugin-cli --plugin-file /usr/share/jenkins/ref/plugins.txt
