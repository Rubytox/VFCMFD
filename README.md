# Prise en main de SURF
## Présentation de l'application

Le programme réalisé ici s'appelle `main` et utilise la bibliothèque OpenCV afin d'appliquer l'algorithme SURF sur des images. Etant données deux images passées en paramètre au programme, l'algorithme :
- calcule les points d'intérêts des deux images ;
- détermine les correspondances entre les points d'intérêts des deux images, et les trace :
    - dans un premier temps le programme prend en paramètres une image et un extrait de cette image ;
    - dans un second temps, ce programme doit ne prendre qu'une seule image en paramètre et trouver les configurations similaires dans l'image. 

Pour le second point, j'ai d'abord commencé par réaliser la première étape de l'algorithme, mais en passant en paramètre deux fois la même image : on trouve ici naturellement beaucoup de correspondances. Cependant, si une partie de l'image a été copiée/déplacée dans l'image, alors l'algorithme devrait également trouver une correspondance entre la zone d'origine et la zone déplacée : ainsi, j'ai commencé par filtrer les correspondances en ne conservant que celles dont le point de départ sur la première image est différent du point d'arrivée de la seconde image ; comme ceci on élimine les correspondances évidentes. 

L'algorithme marche sur des exemples très simples et grossiers, à priori sans appliquer de transformations type rotation/homothétie. 

## Compilation de l'application

L'algorithme SURF est un algorithme propriétaire, et OpenCV ne l'inclut plus de base. Il a donc fallu que je recompile la bibliothèque pour pouvoir m'en servir. Pour ce faire, j'ai téléchargé les sources de OpenCV ainsi que le repository github de `opencv_contrib`. Ensuite, on se place dans le répertoire des sources de OpenCV et on exécute :
```
$ cd sources_opencv
$ mkdir build
$ cd build
$ cmake -DOPENCV_ENABLE_NONFREE:BOOL=ON -DOPENCV_EXTRA_MODULES_PATH=<chemin_vers_opencv_contrib>/modules ..
$ make -j$(nproc)
$ sudo make install
```

La ligne 4 est particulièrement importante : si on ne met pas le flag `OPENCV_ENABLE_NONFREE` à vrai, on aura compilé la bibliothèque mais les fonctionnalités propriétaires ne seront pas activées. 

Une fois ceci fait, on devrait disposer de la bibliothèque OpenCV4. On peut ensuite compiler l'application de la manière suivante :
```
Dans le dossier Week1/images
$ cd build
$ cmake ..
$ make
```

On lance alors l'application avec :
```
$ build/main extrait.jpg image_source.jpg
```
Une fenêtre devrait apparaître, contenant les deux images et les lignes représentant les correspondances trouvées. On appuie alors sur la touche `s` pour sauvegarder l'image, qui sera sauvegardée sous le nom `extrait_keypoints.jpg`. 

## Paramètres modifiables de l'algorithme
Le plus gros du code se passe principalement dans la fonction `surf_matching` qui prend les deux images en paramètres ainsi que la valeur minimale du seuil Hessien : plus le seuil est grand, plus les points d'intérêts sont sélectifs et donc on a moins de correspondances à la fin, mais elles sont normalement plus précises. Dans le code actuel, le seuil est mis à 300.

Ensuite, si on veut passer du mode "deux images et correspondances" au mode "correspondances sur une seule image", il y a du code à commenter/décommenter : je pense que je mettrai plutôt un argument en plus à passer au programme afin de décider quel mode utiliser, mais ce n'est pas encore fait.
