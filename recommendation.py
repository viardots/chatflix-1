# coding: utf-8

from User import User
from random import randint
from random import choice

import numpy as np
from sklearn.cluster import KMeans

from movielens import load_movies, load_simplified_ratings, load_ratings


class Recommendation:

    def __init__(self):

        # Importe la liste des films
        # Dans la variable 'movies' se trouve la correspondance entre l'identifiant d'un film et le film
        # Dans la variables 'movies_list' se trouve les films populaires qui sont vus par les utilisateurs
        self.movies = load_movies()
        self.movies_dict = { m.id : m for m in self.movies}
        self.movies_id = {m.id : i for i,m in enumerate(self.movies)}
        # Construit un dictionnaire de genre
        self.vect_genre=[]
        for m in self.movies:
            vect= [ m.unknown,
                    m.action,
                    m.adventure,
                    m.animation,
                    m.children,
                    m.comedy,
                    m.crime,
                    m.documentary,
                    m.drama,
                    m.fantasy,
                    m.film_noir,
                    m.horror,
                    m.musical,
                    m.mystery,
                    m.romance,
                    m.sci_fi,
                    m.thriller,
                    m.war,
                    m.western]
            self.vect_genre.append(vect)
        self.vect_genre = np.array(self.vect_genre)
        self.kmeans = KMeans(n_clusters=10)
        self.fit = self.kmeans.fit_predict(self.vect_genre)
        self.movies_list = []


        # Importe la liste des notations
        # Dans le tableau 'ratings' se trouve un objet avec un attribut 'movie' contenant l'identifiant du film, un
        # attribut 'user' avec l'identifiant de l'utilisateur et un attribut 'is_appreciated' pour savoir si oui ou non
        # l'utilisateur aime le film
        #self.ratings = load_simplified_ratings()
        self.ratings = load_ratings()

        # Les utilisateurs du fichier 'ratings-popular-simplified.csv' sont stockés dans 'test_users'
        self.test_users = {}
        # Les utilisateurs du chatbot facebook seront stockés dans 'users'
        self.users = {}

        # Lance le traitement des notations
        self.process_ratings_to_users()

        self.user_cluster_matrix = [self.process_normal_cluster(u) for u in self.test_users.values()]


    # Traite les notations
    # Crée un utilisateur de test pour chaque utilisateur dans le fichier
    # Puis lui attribue ses films aimés et détestés
    def process_ratings_to_users(self):
        for rating in self.ratings:
            user = self.register_test_user(rating.user)
            if rating.is_appreciated is not None:
                if rating.is_appreciated:
                    user.good_ratings.append(rating.movie)
                else:
                    user.bad_ratings.append(rating.movie)
            elif rating.score is not None:
                user.ratings[rating.movie] = int(rating.score)
            self.movies_list.append(rating.movie)

    # Enregistre un utilisateur de test s'il n'existe pas déjà et le retourne
    def register_test_user(self, sender):
        if sender not in self.test_users.keys():
            self.test_users[sender] = User(sender)
        return self.test_users[sender]

    # Enregistre un utilisateur s'il n'existe pas déjà et le retourne
    def register_user(self, sender):
        if sender not in self.users.keys():
            self.users[sender] = User(sender)
        return self.users[sender]

    # Retourne les films aimés par un utilisateur
    def get_movies_from_user(self, user):
        movies_list = []
        good_movies = user.good_ratings
        for movie_number in good_movies:
            movies_list.append(self.movies[movie_number].title)
        return movies_list

    # Calcule vecteur normalise pour un utilisateur
    def process_normal_cluster(self, user):
        # Cluster vide
        clusters = [[] for _ in range(10)]
        rate_mean = 0
        rate_count = 0
        for m, rate in user.ratings.items():
            cluster_movie = self.fit[self.movies_id[m]]
            clusters[cluster_movie].append(rate)
            rate_mean += rate
            rate_count += 1
        if rate_count!=0:
            rate_mean = rate_mean / rate_count
        clusters_mean = [(np.mean(rates) - rate_mean) if rates else 0 for rates in clusters]
        user.cluster_prefs = clusters_mean
        return clusters_mean

    # Calcule la note moyenne d'un groupe d'utilisateurs pour un film donné
    def rate_movie(self, movie, users):
        rates_movie =[ u.ratings[movie.id] for u in users if movie.id in u.ratings.keys()]
        return np.mean(rates_movie) if rates_movie else 0
    # Affiche la recommandation pour l'utilisateur
    def make_recommendation(self, user):
        self.process_normal_cluster(user)
        # Similarité d'utilisateurs
        similar_users = self.compute_all_similarities(user)
        similar_users.sort(key=lambda ur: ur[1], reverse=True)
        similar_users = [u for u,_ in similar_users[0:50]]
        rates_movies = [(m.id,self.rate_movie(m,similar_users)) for m in self.movies]
        rates_movies.sort(key=lambda rm: rm[1], reverse=True)
        rates_movie=[m for m,rate in rates_movies if m not in user.ratings.keys()]
        recommended_movies = [self.movies_dict[f].title for f in rates_movie[0:5]]
        return ";".join(recommended_movies)

    # Pose une question à l'utilisateur
    def ask_question(self, user):
        identifiant_alea = choice(self.movies_list)
        film_demande = self.movies_dict[identifiant_alea]
        user.set_question(identifiant_alea)
        return "Avez vous aimé le film " + film_demande.title + "(répondez par oui ou non ou mettez une note de 1 à 5)"

    # Calcule la similarité entre 2 utilisateurs
    @staticmethod
    def get_similarity(user_a, user_b):
        prod_scalaire = sum([user_a.cluster_prefs[i]*user_b.cluster_prefs[i] for i in range(len(user_a.cluster_prefs))])
        norm_user_a = sum([rate**2 for rate in user_a.cluster_prefs])
        norm_user_b = sum([rate ** 2 for rate in user_b.cluster_prefs])
        if prod_scalaire == 0 :
            return 0
        return prod_scalaire/(norm_user_a*norm_user_b)

    # Calcule la similarité entre un utilisateur et tous les utilisateurs de tests
    def compute_all_similarities(self, user):
        return [(u,self.get_similarity(user,u)) for u in self.test_users.values()]