# coding: utf-8

from User import User
from random import randint
from random import choice

import numpy as np
from sklearn.cluster import KMeans

from movielens import load_movies, load_simplified_ratings


class Recommendation:

    def __init__(self):

        # Importe la liste des films
        # Dans la variable 'movies' se trouve la correspondance entre l'identifiant d'un film et le film
        # Dans la variables 'movies_list' se trouve les films populaires qui sont vus par les utilisateurs
        self.movies = load_movies()
        self.movies_dict = { m.id : m for m in self.movies}
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
        self.kmeans.fit_predict(self.vect_genre)

        self.movies_list = []

        # Importe la liste des notations
        # Dans le tableau 'ratings' se trouve un objet avec un attribut 'movie' contenant l'identifiant du film, un
        # attribut 'user' avec l'identifiant de l'utilisateur et un attribut 'is_appreciated' pour savoir si oui ou non
        # l'utilisateur aime le film
        self.ratings = load_simplified_ratings()

        # Les utilisateurs du fichier 'ratings-popular-simplified.csv' sont stockés dans 'test_users'
        self.test_users = {}
        # Les utilisateurs du chatbot facebook seront stockés dans 'users'
        self.users = {}

        # Lance le traitement des notations
        self.process_ratings_to_users()

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
                user.ratings.append(rating)
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

    # Affiche la recommandation pour l'utilisateur
    def make_recommendation(self, user):
        similar_users = self.compute_all_similarities(user)
        similar_users.sort(key=lambda ur: ur[1], reverse=True)
        if len(similar_users)>4:
            prefered_movies = { f : 1 for f in similar_users[0][0].good_ratings}
            for u in similar_users[1:5]:
                for f in u.good_ratings:
                    if f in prefered_movies.keys():
                        prefered_movies[f] += 1
                    else:
                        prefered_movies[f] = 1
            prefered_movies = sorted([{"movie":f, "rate":nb} for f,nb in prefered_movies ],
                                     key=lambda film_rate: film_rate[1],
                                     reverse=True)
            recommended_movies = [self.movies_dict[f["movie"]].title for f in prefered_movies if f["rate"]>1]
            return ";".join(recommended_movies)
        elif len(similar_users)>0:
            similar_user = similar_users[0][0]
            recommended_movies = [self.movies_dict[f].title for f in similar_user.good_ratings]
            return ";".join(recommended_movies)
        return "Vous n'avez pas de recommandation pour le moment."

    # Pose une question à l'utilisateur
    def ask_question(self, user):
        identifiant_alea = choice(self.movies_list)
        film_demande = self.movies_dict[identifiant_alea]
        user.set_question(identifiant_alea)
        return "Avez vous aimé le film " + film_demande.title

    # Calcule la similarité entre 2 utilisateurs
    @staticmethod
    def get_similarity(user_a, user_b):
        score = 0
        for film_id in user_a.good_ratings:
            if film_id in user_b.good_ratings:
                score += 1
            elif film_id in user_b.bad_ratings:
                score -= 1
        for film_id in user_a.bad_ratings:
            if film_id in user_b.good_ratings:
                score -= 1
            elif film_id in user_b.bad_ratings:
                score += 1
        for film_id in user_a.neutral_ratings:
            if film_id in user_b.neutral_ratings:
                score += 1
        return score / user_a.get_norm()

    # Calcule la similarité entre un utilisateur et tous les utilisateurs de tests
    def compute_all_similarities(self, user):
        return [(u,self.get_similarity(user,u)) for u in self.users.values()]
