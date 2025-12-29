import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class RecommendationSystem:
    """Recommendation system using collaborative filtering and content-based filtering"""
    
    def __init__(self):
        self.items_df = None
        self.user_ratings = None
        self.similarity_matrix = None
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        
    def load_items_data(self, items_data):
        """Load items data (movies, books, products)"""
        self.items_df = pd.DataFrame(items_data)
        return self.items_df
    
    def load_ratings_data(self, ratings_data):
        """Load user ratings data"""
        self.user_ratings = pd.DataFrame(ratings_data)
        return self.user_ratings
    
    def content_based_recommendations(self, item_id, num_recommendations=5):
        """Generate content-based recommendations"""
        if self.items_df is None:
            return []
        
        # Create TF-IDF matrix for item descriptions
        descriptions = self.items_df.get('description', self.items_df.get('genre', []))
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(descriptions)
        
        # Calculate similarity matrix
        self.similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Get similar items
        if item_id >= len(self.similarity_matrix):
            return []
        
        similar_items = self.similarity_matrix[item_id]
        top_indices = np.argsort(similar_items)[::-1][1:num_recommendations+1]
        
        recommendations = self.items_df.iloc[top_indices].to_dict('records')
        return recommendations
    
    def collaborative_filtering_recommendations(self, user_id, num_recommendations=5):
        """Generate recommendations using collaborative filtering"""
        if self.user_ratings is None or len(self.user_ratings) == 0:
            return []
        
        # Create user-item rating matrix
        user_item_matrix = self.user_ratings.pivot_table(
            index='user_id', 
            columns='item_id', 
            values='rating',
            fill_value=0
        )
        
        # Calculate user similarity
        user_similarity = cosine_similarity(user_item_matrix)
        
        if user_id >= len(user_similarity):
            return []
        
        # Find similar users
        similar_users = user_similarity[user_id]
        similar_user_indices = np.argsort(similar_users)[::-1][1:6]
        
        # Get items rated by similar users
        recommendations = {}
        for similar_user_idx in similar_user_indices:
            rated_items = user_item_matrix.iloc[similar_user_idx]
            for item_id, rating in rated_items.items():
                if rating > 0:
                    if item_id not in recommendations:
                        recommendations[item_id] = []
                    recommendations[item_id].append(rating)
        
        # Calculate average rating for each item
        item_scores = {item_id: np.mean(ratings) for item_id, ratings in recommendations.items()}
        top_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:num_recommendations]
        
        return [{'item_id': item_id, 'predicted_rating': score} for item_id, score in top_items]
    
    def hybrid_recommendations(self, user_id, item_id, num_recommendations=5):
        """Generate hybrid recommendations combining content and collaborative filtering"""
        content_recs = self.content_based_recommendations(item_id, num_recommendations)
        collab_recs = self.collaborative_filtering_recommendations(user_id, num_recommendations)
        
        # Combine recommendations
        combined = {}
        for rec in content_recs:
            rec_id = rec.get('id', rec.get('item_id', 0))
            combined[rec_id] = combined.get(rec_id, 0) + 1
        
        for rec in collab_recs:
            rec_id = rec.get('item_id')
            if rec_id:
                combined[rec_id] = combined.get(rec_id, 0) + rec.get('predicted_rating', 0)
        
        # Sort by combined score
        sorted_recs = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:num_recommendations]
        return sorted_recs

def main():
    print('Recommendation System')
    print('Supports: Collaborative Filtering, Content-Based, and Hybrid approaches')
    
    # Initialize recommender
    recommender = RecommendationSystem()
    print('\nRecommendation system initialized and ready.')

if __name__ == '__main__':
    main()
