import json

"""
Authors:
    Oliwier Kossak s22018
    Daniel Klimowski S18504
    
Description of the problem:
    Creating a film engine that will display a list of recommended and non-recommended films to the given user.

Instructions for use:
    1. Required json file with users rated movies
    2. Enter name of user that you want display list of recommended and not recommended movies (name of user must be in json file).
    
Command engine operation:
    1. Needed json file that contains users names and movies title with rating.
    2. Provide name of user whom movies will be recommended and not recommended.
    3. The program searches for the same titles of rated movies for user and other users.
    4. The program calculates euclidean distance for same titles of rated movies (There must be at least 3 rated films 
       with the same title to calculate the Euclidean distance).
    5. The best match user is user with the lowest value of euclidean distance.
    6. The program creates a sorted list of movies of the best-matched user.
    7. From the best-matched user is displayed recommended movies and not recommended (titles of movie that are rated of both user are not displayed)"""


class MovieEngine:

    def __init__(self, target_user, data_path, number_of_recommendation=5, verbose=True):
        """
        Engine Initialization

        Parameters:
            target_user (str): Name of user that recommediaton of movies will be display.
            data_path (str):  Path to json file with users names and  rated movies.
            number_of_recommendation (int): Number of recommended and not recommended movies that will be displayed.
            verbose (bool): Display information about  process of compare similarity with other users.
        """
        self.target_user = target_user
        self.data_path = data_path
        self.number_of_recommendation = number_of_recommendation
        self.verbose = verbose

    def calculate_euclidean_distance(self):
        """
        The function that assessing the similarity of two users.

        Returns:
            dictionary: Dictionary include euclidean distances of compare target user with every other user in json file.
        """
        same_movies = self.search_for_same_movies()
        all_movies = self.read_json_file(self.data_path)
        euclidean_distance_users = {}

        for user in same_movies:
            distance_sum = 0
            for movie in same_movies[user]:
                if movie in all_movies[self.target_user]:
                    distance_sum += (all_movies[user][movie] - all_movies[self.target_user][movie]) ** 2
            euclidean_distance = distance_sum ** (1 / 2)
            euclidean_distance_users[user] = euclidean_distance
        return euclidean_distance_users

    def search_for_same_movies(self):
        """
        The function that finds these collaboratively rated movies for the target user and other users.

        Returns:
            dictionary: A dictionary with keys as other usernames and values that are collectively rated by the movies.
        """
        json_file = self.read_json_file(f'{self.data_path}')
        same_movies = {}

        for user in json_file:
            same_movies_for_user = {}
            count_movies = 0
            for movie in json_file[user]:
                if movie in json_file[self.target_user] and self.target_user != user:
                    count_movies += 1
                    same_movies_for_user[movie] = json_file[user][movie]
            if count_movies > 2:
                same_movies[user] = same_movies_for_user

        return same_movies

    def read_json_file(self, data_path):
        """
        The function to read json file with users and rated movies.

        Returns:
            dictionary: users rated movies.
        """
        with open(f'{data_path}') as json_file:
            data = json.load(json_file)
        return data

    def sort_movies(self):

        user = self.search_for_best_match_user()
        movies = self.read_json_file(self.data_path)
        user_movies = movies[user]
        movies_list = [(k, v) for k, v in user_movies.items()]
        movies_list.sort(key=lambda s: s[1])
        return movies_list

    def search_for_best_match_user(self):
        """
        The function that search for most similar user for target user.

        Return:
            str: Name of best matched user.
        """
        users = self.calculate_euclidean_distance()
        users_list = [(k, v) for k, v in users.items()]
        users_list.sort(key=lambda s: s[1])
        best_match_user = users_list[0][0]
        return best_match_user

    def movie_suggestions(self):
        """
        The function returns list of recommended and not recommended movies.

        Return:
             (list,list): returns list of recommended and not recommended movies.
        """
        movies = self.sort_movies()
        same_movies = self.search_for_same_movies()
        recommended_movies = []
        not_recommended_movies = []
        count_recommended_movies = 1
        count_not_recommended_movies = 0

        while len(not_recommended_movies) < self.number_of_recommendation and len(
                recommended_movies) < self.number_of_recommendation:
            movie = [movie for v in same_movies.values() for movie in v.keys()]
            if movies[-count_recommended_movies][0] not in movie and len(
                    recommended_movies) < self.number_of_recommendation:
                recommended_movies.append(movies[-count_recommended_movies])
            if movies[count_not_recommended_movies][0] not in movie and len(
                    not_recommended_movies) < self.number_of_recommendation:
                not_recommended_movies.append(movies[count_not_recommended_movies])
            count_recommended_movies += 1
            count_not_recommended_movies += 1

        return recommended_movies, not_recommended_movies

    def display_similar_movies(self):
        """The function displays collaboratively rated movies of target user and other users"""

        all_movies = self.read_json_file(self.data_path)
        matches_movies = self.search_for_same_movies()
        euclides_distance = self.calculate_euclidean_distance()

        for user in matches_movies:
            user_same_movies = {}
            for movie in matches_movies[user]:
                user_same_movies[movie] = all_movies[self.target_user][movie]
            print(f'{self.target_user} COMMON MOVIES WITH {user}')
            print(user_same_movies)
            print(matches_movies[user])
            print(f'Euclides distance: {round(euclides_distance[user], 2)}')
            print()

    def display_recommendation(self):
        """The function that display recommended and not recommended movies for target user"""
        recommended_movies, not_recommended_movies = self.movie_suggestions()
        recommended_user = self.search_for_best_match_user()

        if self.verbose:
            self.display_similar_movies()

        print(f'{recommended_user} not recommended:')
        for movie in not_recommended_movies:
            print(f'Movie: {movie[0]} , Rate: {movie[1]}')
        print()
        print(f'{recommended_user} recommended:')
        for movie in recommended_movies:
            print(f'Movie: {movie[0]} , Rate: {movie[1]}')


user_name = input("Enter user name: ")
engine = MovieEngine(f'{user_name}', 'data.json')
engine.display_recommendation()
