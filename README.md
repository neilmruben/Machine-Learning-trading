# Machine-Learning-trading


Machine learning & trading : Dashboard interactif en Python avec Streamlit
Ce dashboard interactif permet de prédire le mouvement d’un actif (haussier / baissier) sélectionné au préalable. 
Il permet de choisir les paramètres des features utilisés ainsi que certains paramètres avant modèlisation comme la taille de l’échantillon de test 
et le nombre de kfold pour la Cross-Validation. Une fois la Cross Validation effectuée on retiens les 3 meilleurs modèles de classification
pour notre prédiction à court-terme (ici une prédiction à intervalle de 2 minutes). 
Il permet une fois les modèles entrainés de télécharger ces modèles au format pickle pour les réutiliser (il est possible de charger 
directement sur le dashboard des fichiers pickle en enlevant les commentaires du code Python disponible sur github).


<img width="946" alt="image" src="https://user-images.githubusercontent.com/81652761/212491063-181bda06-6b69-4cc3-b6d8-e168caf0ab76.png">


<img width="530" alt="image" src="https://user-images.githubusercontent.com/81652761/212491098-222948da-1253-4077-9d33-36ace5851126.png">
