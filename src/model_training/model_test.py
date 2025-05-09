#  PredictionBot
#  Copyright (C) 2025 CatraMyBeloved
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Load models and preprocessor
try:
    print("Loading models and preprocessor...")
    preprocessor = joblib.load('./models/preprocessor.pkl')
    rf_model = joblib.load('./models/random_forest.pkl')
    et_model = joblib.load('./models/extra_trees.pkl')
    nn_model = joblib.load('./models/neural_network.pkl')
    ensemble_model = joblib.load('./models/ensemble.pkl')
    print("All models loaded successfully!")

except FileNotFoundError as e:
    print(f"Error loading models: {e}")
    print("Make sure you've saved the models in the 'models' directory.")
    exit()


# 2. Create prediction function
def predict_match(team1_name, team2_name, map_name, ban1_hero=None,
                  ban2_hero=None):
    """Make predictions using all loaded models"""

    # Create input data
    input_data = pd.DataFrame({
        'team_name': [team1_name],
        'team_name_opp': [team2_name],
        'map_name': [map_name],
        'banned_hero': [ban1_hero if ban1_hero else 'No Ban'],
        'banned_hero_opp': [ban2_hero if ban2_hero else 'No Ban']
    })

    # Preprocess input data
    input_transformed = preprocessor.transform(input_data)

    # Get predictions from all models
    predictions = {
        'Random Forest': rf_model.predict_proba(input_transformed)[0][1],
        'Extra Trees': et_model.predict_proba(input_transformed)[0][1],
        'Neural Network': nn_model.predict_proba(input_transformed)[0][1],
        'Ensemble': ensemble_model.predict_proba(input_transformed)[0][1]
    }

    return predictions


# 3. Create some test cases
test_cases = [
    {'team1_name': "Team Falcons", "team2_name": "Virtus.Pro", 'map_name':
        "King's Row",
     'ban1': "Ana",
     'ban2': "Tracer"},
    {'team1_name': "NTMR", 'team2_name': "Spacestation Gaming", 'map_name':
        "Ilios",
     'ban1':
        "Mercy",
     'ban2': "Reinhardt"},
    {'team1_name': "Team Peps", 'team2_name': "The Ultimates", 'map_name':
        "Busan", 'ban1':
        "Ana",
     'ban2': "Genji"}
]

# 4. Run predictions on test cases
print("\nRunning test predictions:\n" + "-" * 50)
for i, case in enumerate(test_cases):
    print(f"\nTest Case {i + 1}:")
    team1_name = case['team1_name']
    team2_name = case['team2_name']
    map_name = case['map_name']
    ban1 = case['ban1']
    ban2 = case['ban2']

    print(f"Match: {team1_name} vs {team2_name} on {map_name}")
    print(f"Bans: {ban1 or 'None'} / {ban2 or 'None'}")

    # Get predictions
    predictions = predict_match(team1_name, team2_name, map_name, ban1, ban2)

    # Print predictions
    print("Model Predictions:")
    for model_name, probability in predictions.items():
        winner = team1_name if probability > 0.5 else team2_name
        confidence = max(probability, 1 - probability)
        print(
            f"  {model_name}: {winner} wins ({probability:.2%} chance for {team1_name})")

    # Show model agreement
    prob_values = list(predictions.values())
    std_dev = np.std(prob_values)
    print(
        f"Model agreement: {'High' if std_dev < 0.05 else 'Medium' if std_dev < 0.1 else 'Low'} (σ={std_dev:.3f})")

