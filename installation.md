Ce document explique comment installer Docker, récupérer l'image Docker `gontary/stage:latest`, configurer l'environnement hôte et lancer un conteneur pour utiliser l'espace de travail ROS Melodic.

## Prérequis

*   Un système d'exploitation Linux (recommandé pour la compatibilité avec ROS et l'affichage graphique).
*   `sudo` ou des privilèges administrateur peuvent être nécessaires pour certaines commandes (installation de Docker, `docker run`).

## Étape 1 : Installation de Docker

Si vous n'avez pas Docker installé, suivez les instructions officielles pour votre distribution Linux.

1.  **Mettre à jour les paquets :**
    ```bash
    sudo apt-get update
    ```
2.  **Installer les dépendances nécessaires :**
    ```bash
    sudo apt-get install \
        apt-transport-https \
        ca-certificates \
        curl \
        gnupg \
        lsb-release
    ```
3.  **Ajouter la clé GPG officielle de Docker :**
    ```bash
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    ```
4.  **Configurer le dépôt stable :**
    ```bash
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    ```
5.  **Installer Docker Engine :**
    ```bash
    sudo apt-get update
    sudo apt-get install docker-ce docker-ce-cli containerd.io
    ```
6.  **(Optionnel mais Recommandé) Gérer Docker en tant qu'utilisateur non-root :**
    Ajoutez votre utilisateur au groupe `docker` pour éviter d'utiliser `sudo` pour chaque commande Docker. 
    ```bash
    sudo groupadd docker
    sudo usermod -aG docker $USER
    newgrp docker # Ou déconnectez-vous et reconnectez-vous
    ```
7.  **Vérifier l'installation :**
    ```bash
    docker run hello-world
    ```
    Si cela fonctionne, Docker est prêt.

## Étape 2 : Préparation de l'Environnement Hôte

1.  **Créer le dossier de l'espace de travail (`danger_ws`) sur votre machine hôte :**
    Ce dossier sera partagé avec le conteneur Docker. Créez-le dans votre répertoire personnel (`~`) ou à l'emplacement de votre choix.
    ```bash
    mkdir ~/danger_ws
    ```
    Ensuite copiez le fichier danger_ws.zip dans le dossier ~/danger_ws et extrayez le fichier
    ```bash
    unzip workspace.zip
    rm workspace.zip
    ```
2.  **Autoriser les connexions locales au serveur X (pour l'affichage graphique GUI) :**
    Cette commande permet aux applications graphiques lancées depuis le conteneur de s'afficher sur votre écran. Exécutez-la dans un terminal sur votre machine hôte.
    ```bash
    xhost +local:docker
    ```

## Étape 3 : Télécharger l'Image Docker

Récupérez l'image Docker `gontary/stage:latest` depuis Docker Hub :
```bash
docker pull gontary/stage:latest
```

## Étape 4 : Lancer le Conteneur Docker

1.  **Définir les chemins de l'espace de travail :**
    Configurez les variables d'environnement pour spécifier où se trouve votre espace de travail sur la machine hôte et où il sera monté dans le conteneur. Remplacez `user_name` par votre nom d'utilisateur réel si vous n'avez pas créé `danger_ws` directement dans `/home/user_name/`.
    ```bash
    export HOST_WS_PATH="$HOME/danger_ws" # Ou le chemin complet vers votre dossier
    export CONTAINER_WS_PATH="/root/danger_ws"
    ```
    *Note : Ces variables sont définies pour la session de terminal actuelle. Vous devrez les redéfinir si vous ouvrez un nouveau terminal.*

2.  **Exécuter le conteneur :**
    Cette commande lance le conteneur en mode interactif (`-it`), lui donne un nom (`--name`), configure le partage d'affichage (`--env DISPLAY`, `--env QT_X11...`, `--volume /tmp/.X11...`), partage les informations d'authentification X (`--env XAUTHORITY`, `--volume $HOME/.Xauthority...`), monte votre dossier de travail local (`--volume "$HOST_WS_PATH:$CONTAINER_WS_PATH:rw"`) et partage le périphérique audio (`--device=/dev/snd`).
    ```bash
    sudo docker run -it \
        --name ros_stage_container \
        --shm-size="1g" \
        --env DISPLAY=$DISPLAY \
        --env QT_X11_NO_MITSHM=1 \
        --env XAUTHORITY=/root/.Xauthority \
        --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
        --volume="$HOME/.Xauthority:/root/.Xauthority:rw" \
        --volume="$HOST_WS_PATH:$CONTAINER_WS_PATH:rw" \
        --device=/dev/snd:/dev/snd \
        gontary/stage:latest \
        bash
    ```

    Vous devriez maintenant être dans une session `bash` à l'intérieur du conteneur Docker, avec l'invite de commande ressemblant à quelque chose comme `root@<container_id>:/#`.

## Étape 5 : Utilisation de l'Environnement ROS Melodic dans le Conteneur

Une fois à l'intérieur du conteneur (après l'étape 4) :

1.  **Naviguer vers l'espace de travail monté :**
    Le dossier `danger_ws` de votre machine hôte est maintenant accessible à `/root/danger_ws` dans le conteneur.
    ```bash
    cd /root/danger_ws
    ```

2.  **Sourcer l'environnement ROS Melodic principal :**
    Cela rend les commandes ROS (`roscore`, `roslaunch`, etc.) disponibles.
    ```bash
    source /opt/ros/melodic/setup.bash
    ```

3.  **Compiler l'espace de travail (si nécessaire) :**
    Si le code source dans `danger_ws/src` a changé ou si c'est la première fois, compilez-le. Assurez-vous d'être dans le répertoire racine de l'espace de travail (`/root/danger_ws`).
    ```bash
    catkin build
    ```

4.  **Sourcer l'environnement local de l'espace de travail :**
    Après une compilation réussie, sourcez le fichier `setup.bash` généré dans le dossier `devel` pour rendre les paquets de votre espace de travail accessibles.
    ```bash
    source devel/setup.bash
    ```
    *Cette étape doit être faite **après** `catkin build` et **après** avoir sourcé `/opt/ros/melodic/setup.bash`.*

5.  **Lancer un fichier de lancement ROS :**
    Exécutez votre fichier de lancement. Remplacez `clf_cbf_qp` par le nom de votre paquet ROS et `nom_du_launch.launch` par le nom exact de votre fichier de lancement.
    ```bash
    roslaunch clf_cbf_qp nom_du_launch.launch
    ```

Vous pouvez maintenant interagir avec votre application ROS comme d'habitude. Les applications avec interface graphique (comme RViz, rqt_plot) devraient s'afficher sur votre bureau hôte grâce à la configuration X11.

## Quitter et Revenir

*   Pour quitter la session `bash` du conteneur, tapez `exit`. Le conteneur s'arrêtera.
*   Pour redémarrer et vous rattacher au *même* conteneur arrêté (en conservant les modifications internes non liées aux volumes) :
    ```bash
    sudo docker start ros_stage_container
    sudo docker exec -it ros_stage_container /bin/bash
    ```

## Étape 6 : Mettre à jour le Dépôt `clf_cbf_qp` (Optionnel)

Si vous avez besoin de récupérer les dernières modifications du dépôt `clf_cbf_qp`, notamment depuis une branche spécifique comme `clf`, suivez ces étapes **à l'intérieur du conteneur Docker** :

1.  **Assurez-vous d'être dans le conteneur :**
    Si vous n'êtes pas déjà dans le conteneur, redémarrez-le et attachez-vous-y :
    ```bash
    # Si le conteneur est arrêté
    sudo docker start ros_stage_container
    # sudo docker exec -it ros_stage_container bash
    ```

2.  **Naviguer vers le répertoire du dépôt :**
    Le code source est monté dans `/root/danger_ws/src`. Allez dans le dossier spécifique du dépôt `clf_cbf_qp`.
    ```bash
    cd /root/danger_ws/src/clf_cbf_qp
    ```
    *Note : Si le dossier `clf_cbf_qp` n'existe pas, vous devrez d'abord cloner le dépôt :*
    ```bash
    # cd /root/danger_ws/src
    # git clone https://github.com/Gontary101/clf_cbf_qp.git
    # cd clf_cbf_qp
    ```

3.  **Vérifier l'état actuel et la branche:**
    ```bash
    git status
    git branch
    ```
    Cela vous montre sur quelle branche vous êtes et si vous avez des modifications locales non commitées.

4.  **Passer sur la branche `clf` :**
    Si vous n'êtes pas déjà sur la branche `clf`, basculez dessus. Si la branche `clf` n'existe pas localement, cette commande la créera et la fera suivre la branche distante `origin/clf`.
    ```bash
    git checkout clf
    ```
    *Si vous rencontrez des problèmes (par exemple, des modifications locales non commitées), vous devrez peut-être les gérer d'abord (`git stash`, `git commit`, ou `git checkout -f clf` pour forcer*

5.  **Récupérer les dernières modifications depuis le dépôt distant :**
    Cette commande télécharge les dernières modifications de la branche `clf` depuis le dépôt distant (`origin`) et tente de les fusionner avec votre copie locale.
    ```bash
    git pull origin clf
    ```
    *Si vous êtes déjà sur la branche `clf` qui suit `origin/clf`, un simple `git pull` pourrait suffire.*

6.  **Gérer les conflits (si nécessaire) :**
    Si `git pull` signale des conflits de fusion (merge conflicts), cela signifie que vos modifications locales (si vous en aviez) entrent en conflit avec les modifications distantes. Vous devrez résoudre ces conflits manuellement dans les fichiers concernés, puis ajouter (`git add <fichier_modifié>`) et valider (`git commit`) la résolution.

7.  **Recompiler l'espace de travail :**
    Après avoir mis à jour le code source, retournez à la racine de l'espace de travail et recompilez.
    ```bash
    cd /root/danger_ws
    source /opt/ros/melodic/setup.bash # Assurez-vous que ROS est sourcé
    catkin build
    ```

8.  **Sourcer l'environnement local mis à jour :**
    Mettez à jour votre environnement pour inclure les changements compilés.
    ```bash
    source devel/setup.bash
    ```

