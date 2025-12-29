# Recommendation System

**Task 4:** Create a recommendation system using collaborative filtering or content-based filtering for movies, books, or products

## Overview

This project implements an intelligent recommendation system that suggests products, movies, or books to users based on their preferences and behavior patterns. The system uses advanced machine learning algorithms to understand user preferences and make accurate recommendations.

## Features

- **Collaborative Filtering**: Recommend items based on similar users' preferences
- **Content-Based Filtering**: Recommend items similar to ones the user has liked
- **Hybrid Approach**: Combines both methods for better accuracy
- **User-Item Matrix**: Tracks user ratings and preferences
- **Similarity Metrics**: Cosine similarity and Euclidean distance calculations
- **Scalable Architecture**: Handles large datasets efficiently

## Algorithms

### Collaborative Filtering
- **User-User Similarity**: Find similar users and recommend their favorite items
- **Item-Item Similarity**: Find similar items and recommend based on user history
- **Matrix Factorization**: Decompose user-item matrix to find latent factors

### Content-Based Filtering
- **Item Features**: Movies (genre, director, year), Books (author, genre, year)
- **User Profiles**: Build profiles from user ratings and interactions
- **Similarity Calculation**: Find items similar to user's past preferences

## Installation

```bash
# Clone the repository
git clone https://github.com/sivadurga-123/recommendation-system.git
cd recommendation-system

# Install dependencies
pip install -r requirements.txt
```

## Required Libraries

```
NumPy>=1.21.0
Pandas>=1.3.0
Scikit-learn>=0.24.0
Scipy>=1.7.0
Matplotlib>=3.4.0
Seaborn>=0.11.0
```

## Usage

```python
from recommender import RecommendationSystem

# Initialize the system
recommender = RecommendationSystem(algorithm='collaborative')

# Load user-item ratings data
recommender.load_data('ratings.csv')

# Train the model
recommender.train()

# Get recommendations for a user
user_id = 42
recommendations = recommender.recommend(user_id, num_recommendations=5)
print(f"Top 5 recommendations for user {user_id}:")
for item_id, score in recommendations:
    print(f"  Item {item_id}: Score {score:.2f}")
```

## Dataset Format

### User-Item Ratings (CSV)
```
user_id,item_id,rating
1,101,5
1,102,4
2,101,5
2,103,3
...
```

### Item Features (CSV)
```
item_id,name,genre,year
101,Movie A,Action,2020
102,Movie B,Comedy,2021
103,Book A,Fiction,2019
...
```

## Model Training

```python
# Collaborative Filtering
recommender.train_collaborative(min_common_items=2)

# Content-Based Filtering
recommender.train_content_based(feature_weights={'genre': 0.5, 'year': 0.3})

# Hybrid Approach
recommender.train_hybrid(cf_weight=0.6, cb_weight=0.4)
```

## Evaluation Metrics

The system uses the following metrics for evaluation:
- **Precision@K**: Proportion of recommended items that are relevant
- **Recall@K**: Proportion of relevant items that were recommended
- **NDCG**: Normalized Discounted Cumulative Gain for ranking quality
- **RMSE**: Root Mean Squared Error for rating predictions

## Example Output

```
User 42 - Top 5 Recommendations:
1. Movie: "Inception" (Score: 4.85)
2. Movie: "Interstellar" (Score: 4.72)
3. Movie: "The Matrix" (Score: 4.68)
4. Movie: "Blade Runner" (Score: 4.55)
5. Movie: "Tenet" (Score: 4.42)
```

## Project Structure

```
recommendation-system/
├── recommender.py          # Main implementation
├── README.md               # This file
├── requirements.txt        # Dependencies
├── data/                   # Sample datasets
│   ├── ratings.csv
│   └── items.csv
├── models/                 # Trained models
├── evaluation/             # Evaluation scripts
└── utils/                  # Helper functions
```

## Demo

[Demo Video Link](https://www.linkedin.com/feed/)

## Performance Optimization

- **Vectorized Operations**: Use NumPy for efficient matrix calculations
- **Sparse Matrices**: Handle sparse user-item matrices efficiently
- **Caching**: Cache similarity calculations to avoid recomputation
- **Batch Processing**: Process multiple recommendations in parallel

## Advanced Features

- [ ] Real-time recommendations with streaming data
- [ ] Cold-start problem handling
- [ ] Context-aware recommendations
- [ ] A/B testing framework
- [ ] Explanation generation for recommendations
- [ ] Cross-domain recommendations

## Handling Edge Cases

### Cold Start Problem
- New users: Recommend popular items until we have enough ratings
- New items: Use content-based features to recommend to similar users

### Sparsity Issues
- Use dimensionality reduction (SVD, NMF)
- Apply regularization to prevent overfitting
- Combine with content-based filtering

## License

MIT License - See LICENSE file for details

## Author

Siva Durga (sivadurga-123)

## References

1. "Recommender Systems" - Aggarwal, C.C.
2. "Collaborative Filtering" - Goldberg, D., et al.
3. "Content-Based Recommendation Systems" - Pazzani, M., Billsus, D.
4. Scikit-learn Recommender System Tutorial

## Troubleshooting

### Common Issues

**Issue:** Memory error with large datasets
- **Solution:** Use sparse matrices and process in batches

**Issue:** Poor recommendation quality
- **Solution:** Adjust algorithm parameters or combine multiple algorithms

**Issue:** Slow computation time
- **Solution:** Cache results or use approximate algorithms

## Contact

For questions or support, please open an issue on GitHub.
