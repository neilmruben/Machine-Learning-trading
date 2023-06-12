# Machine-Learning-trading


### Machine learning & trading : Dashboard interactif en Python avec Streamlit

Lien vers le dashboard : https://neilmruben-machine-learning-trading-streamlit-app-gvgls7.streamlit.app/

Ce dashboard interactif permet de prédire le mouvement d’un actif (haussier / baissier) sélectionné au préalable. 
Il permet de choisir les paramètres des features utilisés ainsi que certains paramètres avant modèlisation comme la taille de l’échantillon de test 
et le nombre de kfold pour la cross-validation. Une fois la cross-validation effectuée on retiens les 3 meilleurs modèles de classification
pour notre prédiction à court-terme (ici une prédiction à intervalle de 2 minutes). 

Il permet une fois les modèles entrainés de télécharger ces modèles au format pickle pour les réutiliser (il est possible de charger 
directement sur le dashboard des fichiers pickle en enlevant les commentaires du code Python disponible sur github).
Les données sont normalisées avant la cross-validation avec la fonction MinMaxScaler de scikit-learn, on divsise chaque valeur par la différence entre la valeur maximale de notre colonne et la valeur minimale :

$$ x' = \frac{x - x_{min}}{x_{max} - x_{min}} $$


<img width="946" alt="image" src="https://user-images.githubusercontent.com/81652761/212491063-181bda06-6b69-4cc3-b6d8-e168caf0ab76.png">


<img width="530" alt="image" src="https://user-images.githubusercontent.com/81652761/212491098-222948da-1253-4077-9d33-36ace5851126.png">

<img width="500" alt="image" src="https://user-images.githubusercontent.com/81652761/212499033-d963ed2a-3b54-448b-b275-6c08949f5643.png">

<img width="533" alt="image" src="https://user-images.githubusercontent.com/81652761/212499755-5cc27f92-ded7-432d-a821-6de4717b275f.png">

<img width="515" alt="image" src="https://user-images.githubusercontent.com/81652761/212499847-ff74bce7-bfc2-4a90-8ac6-ef80424d9df7.png">





![_5325a22d-b2bf-40ca-9491-5f4e22d8aecd](https://github.com/neilmruben/Machine-Learning-trading/assets/81652761/185bbbbd-97cc-4a84-9cc4-7d10a10211d7)
