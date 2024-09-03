import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import warnings
import plotly.express as px
# Rajout le 29/08/24 pour la partie Statistiques
import pylab
import scipy.stats as stats
from scipy.stats import shapiro
import pickle
import json



# Pour éviter les messages d'avertissement
warnings.filterwarnings('ignore')

# Charger les données avec cache pour améliorer les performances
@st.cache_data
def load_data():
    urls = {
        "etablissement": 'https://raw.githubusercontent.com/ChristopheMontoriol/French_Industry_Janv24/main/data/base_etablissement_par_tranche_effectif.csv',
        "geographic": 'https://raw.githubusercontent.com/ChristopheMontoriol/French_Industry_Janv24/main/data/name_geographic_information.csv',
        "salaire": 'https://raw.githubusercontent.com/ChristopheMontoriol/French_Industry_Janv24/main/data/net_salary_per_town_categories.csv'
    }
    etablissement = pd.read_csv(urls['etablissement'], sep=',')
    geographic = pd.read_csv(urls['geographic'], sep=',')
    salaire = pd.read_csv(urls['salaire'], sep=',')
    return etablissement, geographic, salaire

etablissement, geographic, salaire = load_data()

# Pré-traitement des données salaire
new_column_names_salaire = {
    'SNHM14': 'salaire',
    'SNHMC14': 'salaire_cadre',
    'SNHMP14': 'salaire_cadre_moyen',
    'SNHME14': 'salaire_employe',
    'SNHMO14': 'salaire_travailleur',
    'SNHMF14': 'salaire_femme',
    'SNHMFC14': 'salaire_cadre_femme',
    'SNHMFP14': 'salaire_cadre_moyen_femme',
    'SNHMFE14': 'salaire_employe_femme',
    'SNHMFO14': 'salaire_travailleur_femme',
    'SNHMH14': 'salaire_homme',
    'SNHMHC14': 'salaire_cadre_homme',
    'SNHMHP14': 'salaire_cadre_moyen_homme',
    'SNHMHE14': 'salaire_employe_homme',
    'SNHMHO14': 'salaire_travailleur_homme',
    'SNHM1814': 'salaire_18-25',
    'SNHM2614': 'salaire_26-50',
    'SNHM5014': 'salaire_+50',
    'SNHMF1814': 'salaire_18-25_femme',
    'SNHMF2614': 'salaire_26-50_femme',
    'SNHMF5014': 'salaire_+50_femme',
    'SNHMH1814': 'salaire_18-25_homme',
    'SNHMH2614': 'salaire_26-50_homme',
    'SNHMH5014': 'salaire_+50_homme'
}

salaire = salaire.rename(columns=new_column_names_salaire)
salaire['CODGEO'] = salaire['CODGEO'].str.lstrip('0').str.replace('A', '0').str.replace('B', '0')

# Configuration de la barre latérale
st.sidebar.title("Sommaire")
pages = ["👋 Intro", "🔍 Exploration des données", "📌Statistiques","📊 Data Visualisation", "🧩 Modélisation", "🔮 Prédiction", "📌 Conclusion"]
page = st.sidebar.radio("Aller vers", pages)

# Affichage de la sélection des données uniquement pour la page "Exploration des données"
if page == pages[1]:
    # Gestion de l'état de la page via session_state
    if 'page' not in st.session_state:
        st.session_state.page = "Etablissement"

    # Sélection de la page de données
    data_pages = ["Etablissement", "Geographic", "Salaire", "Population"]
    # st.sidebar.markdown("### Choix des données")
    st.session_state.page = st.sidebar.selectbox("Sélection du Dataframe", data_pages, index=data_pages.index(st.session_state.page))

st.sidebar.markdown(
    """
    - **Cursus** : Data Analyst
    - **Formation** : Formation Continue
    - **Mois** : Janvier 2024
    - **Groupe** : 
        - Christophe MONTORIOL
        - Issam YOUSR
        - Gwilherm DEVALLAN
        - Yacine OUDMINE
    """
)

# Définition des styles
st.markdown("""
    <style>
        h1 {color: #4629dd; font-size: 70px;}
        h2 {color: #440154ff; font-size: 50px;}
        h3 {color: #27dce0; font-size: 30px;}
        body {background-color: #f4f4f4;}
    </style>
""", unsafe_allow_html=True)

# Page d'introduction
if page == pages[0]:
    st.header("👋 Introduction")
    st.caption("""**Cursus** : Data Analyst | **Formation** : Formation Continue | **Mois** : Janvier 2024 """)
    st.caption("""**Groupe** : Christophe MONTORIOL, Issam YOUSR, Gwilherm DEVALLAN, Yacine OUDMINE""")
     # Ajouter l'image du bandeau
    st.image('https://raw.githubusercontent.com/ChristopheMontoriol/French_Industry_Janv24/main/data/Bandeau_FrenchIndustry.png', use_column_width=True)
    st.write("""
        L’objectif premier de ce projet est d’étudier les inégalités salariales en France. 
        À travers plusieurs jeux de données et plusieurs variables (géographiques, socio-professionnelles, démographiques ...).       
        
        Il sera question dans ce projet de mettre en lumière les facteurs d’inégalités les plus déterminants et de recenser ainsi les variables qui ont un impact significatif sur les écarts de salaire.
        
        En plus de distinguer les variables les plus déterminantes sur les niveaux de revenus, l’objectif de cette étude sera de construire des clusters ou des groupes de pairs basés sur les niveaux de salaire similaires.
        
        Enfin, un modèle de Machine Learning sera entrainé pour prédire au mieux le salaire net moyen en fonction des variables disponibles dans les jeux de données.
    """)




# Page d'exploration des données
elif page == pages[1]:
    st.header("🔍 Exploration des Données")

    # Fonction pour afficher les informations des DataFrames
    def afficher_info(dataframe, name):
        st.write(f"### {name}")
        
        # Calcul des informations demandées
        nb_lignes = dataframe.shape[0]
        nb_colonnes = dataframe.shape[1]
        nb_doublons = dataframe.duplicated().sum()
        nb_donnees_manquantes = dataframe.isna().sum().sum()
        
        # Affichage des informations calculées
        st.write(f"**Nombre de lignes :** {nb_lignes}")
        st.write(f"**Nombre de colonnes :** {nb_colonnes}")
        st.write(f"**Nombre de doublons :** {nb_doublons}")
        st.write(f"**Nombre de données manquantes :** {nb_donnees_manquantes}")
        
        st.write("#### Aperçu des premières lignes de ce jeu de données")
        st.write(dataframe.head())
        
        st.write("#### Informations principales de ce jeu de données")
        buffer = io.StringIO()
        dataframe.info(buf=buffer)
        st.text(buffer.getvalue())
        
        st.write("#### Résumé Statistique du jeu de données")
        st.write(dataframe.describe())

    # Affichage des informations en fonction de la page sélectionnée
    if st.session_state.page == "Etablissement":
        afficher_info(etablissement, "Etablissement")
    elif st.session_state.page== "Geographic":
        afficher_info(geographic, "Geographic")
    elif st.session_state.page == "Salaire":
        afficher_info(salaire, "Salaire")
    elif st.session_state.page == "Population":
        # Afficher un message pour la page Population
        st.write("Pas d'import du dataframe Population, ce jeu de données n'est pas utilisé dans notre projet.")
        # Ajouter un lien vers l'image population.jpg
        st.image('https://raw.githubusercontent.com/ChristopheMontoriol/French_Industry_Janv24/main/data/Population.jpg', use_column_width=True)

# Page de Statistiques
elif page == pages[2]:
    st.header("📊 Statistiques")


   # Test de normalité de Shapiro-Wilk
    stat, p = shapiro(salaire['salaire_cadre_femme'])
    st.write('Test de normalité de Shapiro-Wilk pour la variable salaire_cadre_femme')
    st.write(f"**Statistique :** {stat:.3f}")
    st.write(f"**p-value :** {p:.5f}")
    st.write('La statistique est proche de 1 mais la valeur de la p-value est égale à 0 ce qui suggère que les données de la variable salaire_cadre_femme ne suivent pas une loi normale')

# Suppression des colonnes non nécessaires
    salaire_corr = salaire.drop(columns=['CODGEO', 'LIBGEO'])

# Création de la matrice de corrélation avec Plotly
    matrix_corr = px.imshow(salaire_corr.corr().round(2), text_auto=True)

# Mise en forme des annotations avec deux chiffres après la virgule
    matrix_corr.update_traces(hoverongaps=False)
    matrix_corr.update_layout(title='Matrice de corrélation des salaires',
                          xaxis=dict(title='Variables'),
                          yaxis=dict(title='Variables'),
                          width=1800,
                          height=800)

# Affichage du graphique avec Streamlit
    st.plotly_chart(matrix_corr)


# Page de Data Visualisation
elif page == pages[3]:
    st.header("📊 Data Visualisation")

    st.subheader("Disparité salariale homme/femme")
    
    # Menu déroulant pour la disparité salariale
    disparite_options = ["Disparité salariale par catégorie socioprofessionnelle", "Disparité salariale par tranche d'âge"]
    disparite_choice = st.selectbox("Sélectionnez une visualisation pour la disparité salariale :", disparite_options)
    
    # Visualisation en fonction du choix de l'utilisateur pour la disparité salariale
    if disparite_choice == disparite_options[0]:
        # Disparité salariale par catégorie socioprofessionnelle
        categories = ['Cadres', 'Cadres moyens', 'Employés', 'Travailleurs']
        disparites = [17.60531468314386, 9.887706605652797, 2.472865187964315, 14.680015141858643]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(categories, disparites, color='skyblue')

        ax.set_title('Disparité salariale par catégorie socioprofessionnelle')
        ax.set_xlabel('Catégorie socioprofessionnelle')
        ax.set_ylabel('Disparité salariale (%)')

        plt.xticks(rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig)

    elif disparite_choice == disparite_options[1]:
        # Disparité salariale par tranches d'âge
        tranches_age = ['18-25 ans', '26-50 ans', 'Plus de 50 ans']
        disparites_age = [4.286591078294969, 11.745237278240928, 20.02852196164705]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(tranches_age, disparites_age, color='lightgreen')
        ax.set_title('Disparité salariale par tranche d\'âge')
        ax.set_xlabel('Tranche d\'âge')
        ax.set_ylabel('Disparité salariale (%)')
        plt.xticks(rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig)

    st.subheader("Comparaison de salaire homme/femme")

    # Menu déroulant pour la comparaison des salaires entre hommes et femmes
    comparaison_options = ["Comparaison par catégorie socioprofessionnelle", "Comparaison par tranche d'âge"]
    comparaison_choice = st.selectbox("Sélectionnez une visualisation pour la comparaison des salaires :", comparaison_options)
    
    # Visualisation en fonction du choix de l'utilisateur pour la comparaison des salaires
    if comparaison_choice == comparaison_options[0]:
        # Boîte à moustaches pour chaque catégorie socioprofessionnelle : Hommes et femmes 
        salaires_hommes = salaire[['salaire_cadre_homme', 'salaire_cadre_moyen_homme', 'salaire_employe_homme', 'salaire_travailleur_homme']]
        salaires_femmes = salaire[['salaire_cadre_femme', 'salaire_cadre_moyen_femme', 'salaire_employe_femme', 'salaire_travailleur_femme']]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Boîte à moustaches pour les salaires des hommes
        ax.boxplot([salaires_hommes[col] for col in salaires_hommes.columns], positions=[1, 2, 3, 4], widths=0.4, labels=salaires_hommes.columns, boxprops=dict(color="blue"))

        # Boîte à moustaches pour les salaires des femmes
        ax.boxplot([salaires_femmes[col] for col in salaires_femmes.columns], positions=[1.4, 2.4, 3.4, 4.4], widths=0.4, labels=salaires_hommes.columns, boxprops=dict(color="red"))

        ax.set_title('Comparaison des salaires entre hommes et femmes pour chaque catégorie socioprofessionnelle')
        ax.set_xlabel('Catégorie socioprofessionnelle')
        ax.set_ylabel('Salaire')
        plt.xticks([1.2, 2.2, 3.2, 4.2], ['Cadre', 'Cadre moyen', 'Employé', 'Travailleur'])
        ax.grid(True)
        st.pyplot(fig)

    elif comparaison_choice == comparaison_options[1]:
        # Boîte à moustaches pour chaque tranche d'âge : Hommes et femmes 
        salaires_hommes = salaire[['salaire_18-25_homme', 'salaire_26-50_homme', 'salaire_+50_homme']]
        salaires_femmes = salaire[['salaire_18-25_femme', 'salaire_26-50_femme', 'salaire_+50_femme']]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Boîte à moustaches pour les salaires des hommes
        ax.boxplot([salaires_hommes[col] for col in salaires_hommes.columns], positions=[1, 2, 3], widths=0.4, labels=salaires_hommes.columns, boxprops=dict(color="blue"))

        # Boîte à moustaches pour les salaires des femmes
        ax.boxplot([salaires_femmes[col] for col in salaires_femmes.columns], positions=[1.4, 2.4, 3.4], widths=0.4, labels=salaires_hommes.columns, boxprops=dict(color="red"))

        ax.set_title("Comparaison des salaires entre hommes et femmes pour chaque tranche d'âge")
        ax.set_xlabel("Tranche d'âge")
        ax.set_ylabel('Salaire')
        plt.xticks([1.2, 2.2, 3.2], ['18-25 ans', '26-50 ans', 'Plus de 50 ans'])
        ax.grid(True)
        st.pyplot(fig)


# Page de Modélisation
elif page == pages[4]:
    st.header("🧩 Modélisation")
    st.subheader("Objectif")
    st.write("Prédire le salaire net moyen en fonction des features.")
    
    with st.expander("Modèles étudiés") :
        st.subheader("Liste des modèles")
        st.write("""
                    Afin de déterminer le plus performant possible, nous avons étudié plusieurs modèles de machine learning:
                    - Régression linéaire
                    - Forêt aléatoire
                    - Clustering
        """)


        st.subheader("Exécution des modèles")
        st.write("""
                    Pour chaque modèle appliqué, nous avons suivi les étapes suivantes :
                    1. Instanciation du modèle.
                    2. Entrainement du modèle sur l'ensemble du jeu d'entraînement X_train et y_train.
                    3. Prédictions sur l'ensemble du jeu de test X_test et y_test.
                    4. Evaluation de la performance des modèles en utilisant les métriques appropriées.
                    5. Interprétation des coefficients pour comprendre l'impact de chaque caractéristique sur la variable cible.
                    6. Optimisation du modèle : variation des paramètres, sélection des features utilisées, discrétisation des valeurs.
                    7. Visualisation et analyse des résultats.
                """)
    with st.expander("Modèle retenu") :
        data = {
        'Modèles': ['Forêt aléatoire sans optimisation', 'Forêt aléatoire avec optimisation',  'Forêt aléatoire avec ratio H/F','Forêt aléatoire avec discrétisation','Régression linéaire 1','Régression linéaire 2'],
        'R² train': [0.9994,0.9441,0.9491,0.9456,0.9993,0.9946],
        'R² test': [0.9977,0.8892,0.9376,0.9140,0.9996,0.9938],
        'MSE test': [0.0117, 0.5903,0.3755,0.4577,0.0022,0.0344],
        'MAE test': [0.0747,0.5250,0.4523,0.5240,0.0377,0.1319],
        'RMSE test': [0.1084,0.7683, 0.6127,0.6765,0.0474,0.1855]
            }
    
            
        # Création du DataFrame
        tab = pd.DataFrame(data)
        tab.index = tab.index #+ 1
        # Trouver l'index de la ligne correspondant à "Forêt aléatoire avec discrétisation"
        rf_index = tab[tab['Modèles'] == 'Forêt aléatoire avec discrétisation'].index
    
        # Appliquer un style personnalisé à la ligne spécifique
        styled_tab = tab.style.apply(lambda x: ['background: #27dce0' if x.name in rf_index else '' for i in x], axis=1)
    
    
        # Afficher le tableau avec le style appliqué
        st.subheader("Synthèse des métriques de performance")
        st.table(styled_tab)
        st.markdown("""
                    ##### Choix du modèle :
                    - Les modèles de régression linaires 1 & 2 font de l'overfitting même après optimisation.                     
                    Ils sont donc disqualifiés.
                    - Critères de choix pour le modèle Forêt aléatoire :                    
                    - Les R² ne montrent pas d'overfitting et sont proches de 0.9.                                
                    - Les erreurs restent acceptables.
                    """)
    
        st.write("#### Modèle retenue : Forêt aléatoire avec discrétisation.")


    with st.expander("Evaluation graphique du modèle") :
        st.subheader("Dispersion des résidus & distributions des résidus")
        image_distrib = "https://zupimages.net/up/24/35/r6ed.png"
        st.image(image_distrib, use_column_width=True)
        st.subheader("Comparaison des predictions VS réelles & QQ plot des résidus")
        image_comparaison = "https://zupimages.net/up/24/35/t9c6.png"
        st.image(image_comparaison, use_column_width=True)
            
        st.markdown("""
                    ##### Conclusions :         
                    - Distributions relativement centrées autour de zéro
                    - Distribution normale des résidus
                    - Très peu de points au dela de +/-2
                    - Les résultats obtenus sont plutot uniformes pour toute la plage des données
                    """)
    
    with st.expander("Features d'importance") :
        st.subheader("Histogramme des Features d'importance")
        image_features_importances = "https://zupimages.net/up/24/35/asi7.png"
        st.image(image_features_importances, use_column_width=True)


# Page de Prédiction
elif page == pages[5]:
    st.header("🔮 Prédiction")
    st.subheader('Prédiction du salaire net moyen')
    
    with st.expander("Correspondance des intervalles") :
        data_inter = {
        'Intervalles': ['0', '1',  '2','3','4'],
        'salaire_cadre_discretise (K€)': ['(15.964, 23.1]','(23.1, 30.2]','(30.2, 37.3]','(37.3, 44.4]','(44.4, 51.5]'],
        'salaire_employe_discretise (K€)': ['(8.691,10.46]','(10.46, 12.22]','(12.22, 13.98]','(13.98, 15.74]','(15.74, 17.5]'],
        'salaire_homme_discretise (K€)': ['(10.358, 18.8]','(18.8, 27.2]','(27.2, 35.6]','(35.6, 44.0]','(44.0, 52.4]'],
        'salaire_+50_discretise (K€)': ['(10.454, 19.78]','(19.78, 29.06]','(29.06, 38.34]','(38.34, 47.62]','(47.62, 56.9]'],
        'salaire_+50_femme_discretise (K€)': ['(9.478, 13.8]','(13.8, 18.1]', '(18.1, 22.4]','(22.4, 26.7]','(26.7, 31.0]']
        }

        # Création du DataFrame
        tab1 = pd.DataFrame(data_inter,index=["A", "B", "C", "D", "E"])
        df_reset = tab1.set_index("Intervalles")
    
        # Afficher le tableau avec le style appliqué
        st.subheader("Tableau des intervalles")
        st.table(df_reset)    


    
    def charger_modele(): 
        # Charger le modèle à partir du fichier Pickle
        with open('modele.pkl', 'rb') as fichier_modele:
            modele = pickle.load(fichier_modele)
        return modele
    
    def charger_min_max():
        # Charger les valeurs min et max des caractéristiques depuis le fichier JSON
        with open('feature_min_max.json', 'r') as json_file:
            min_max_dict = json.load(json_file)
        return min_max_dict
    
    def charger_target_mapping():
        # Charger le mapping des targets depuis le fichier JSON
        with open('target_encoding.json', 'r') as json_file:
            target_mapping = json.load(json_file)
        # Convertir les clés en entiers
        target_mapping = {int(key): value for key, value in target_mapping.items()}
        return target_mapping
    
    # Charger les valeurs min et max
    min_max_dict = charger_min_max()

    
    # Créer des curseurs pour chaque caractéristique en utilisant les noms et valeurs depuis le JSON
    caracteristiques_entree = []
    for feature, limits in min_max_dict.items():
        caracteristique = st.slider(
            f"{feature}", 
            float(limits['min']), 
            float(limits['max']), 
            float((limits['min'] + limits['max']) / 2),
            step = 1.0,
        )
        caracteristiques_entree.append(caracteristique)

    data_pred = {
            'Variables': ['salaire_cadre_discretise','salaire_employe_discretise','salaire_homme_discretise','salaire_+50_discretise','salaire_+50_femme_discretise','salaire'],
            'Prédiction N°1': [1,0,0,0,0,13.7],
            'Prédiction N°2': [1,1,1,1,1,18.0]  }
                  
    
    st.write("")
    
    if st.checkbox("Cas concret de prédiction"):
            #st.write("##### Cas concret de prédiction :")
            tab = pd.DataFrame.from_dict(data_pred, orient='index')
            # Définir les colonnes en utilisant la première ligne du DataFrame
            tab.columns = tab.iloc[0]
            # Exclure la première ligne du DataFrame
            tab = tab[1:]    
            st.table(tab)
        
    
        
    # Charger le modèle et le mapping de la cible
    modele = charger_modele()
    target_mapping = charger_target_mapping()
    
    # Préparer les caractéristiques pour la prédiction
    caracteristiques = np.array([caracteristiques_entree])
    
    # Prévoir la classe avec le modèle
    prediction_encoded = modele.predict(caracteristiques)
    
    st.markdown(
    f"<p style='font-size:24px; font-weight:bold;'>La prédiction du salaire net moyen est : {prediction_encoded[0].round(2)}</p>", 
    unsafe_allow_html=True
)

    # Décoder la prédiction
    # prediction_decoded = target_mapping[prediction_encoded[0]]
    
    # Afficher la prédiction
    st.markdown(
        "{prediction_encoded}", 
        unsafe_allow_html=True
    )


# Page de Conclusion
elif page == pages[6]:
    st.header("📌 Conclusion")
    st.write("""Ce projet a été une formidable opportunité de mettre en pratique l'ensemble des compétences acquises durant notre formation. 
    Il nous a permis de développer une approche rigoureuse et méthodique de l'analyse de données, 
    de perfectionner nos compétences techniques, 
    et d'améliorer nos capacités à transformer des données brutes en informations exploitables et pertinentes.""")
    st.write("Nous souhaitons remercier chaleureusement notre mentor, Tarik Anouar, pour nous avoir aidé sur ce projet.")   


