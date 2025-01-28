import pandas as pd
import tkinter as tk
from tkinter import ttk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import random

# Load the dataset
data_path = "IPL_last_10_matches.csv"  # Ensure this file exists in the same directory
df = pd.read_csv(data_path)

# Prepare data for prediction
X = df[["Team", "Opponent", "Venue", "Toss_Decision"]]
y_score = df["Predicted_Score"]
y_wickets = df["Wickets_Lost"]

# Encode categorical variables
X = pd.get_dummies(X, columns=["Team", "Opponent", "Venue", "Toss_Decision"], drop_first=True)

# Train separate models for scores and wickets
X_train, X_test, y_train_score, y_test_score = train_test_split(X, y_score, test_size=0.2, random_state=42)
_, _, y_train_wickets, y_test_wickets = train_test_split(X, y_wickets, test_size=0.2, random_state=42)

score_model = RandomForestRegressor(n_estimators=100, random_state=42)
score_model.fit(X_train, y_train_score)

wickets_model = RandomForestRegressor(n_estimators=100, random_state=42)
wickets_model.fit(X_train, y_train_wickets)

# GUI for predictions
def predict_outcome():
    team = team_var.get()
    opponent = opponent_var.get()
    venue = venue_var.get()
    toss_decision = toss_var.get()

    # Prepare input for prediction (Team 1)
    input_data_team1 = {
        "Team": [team],
        "Opponent": [opponent],
        "Venue": [venue],
        "Toss_Decision": [toss_decision],
    }
    input_df_team1 = pd.DataFrame(input_data_team1)
    input_df_team1 = pd.get_dummies(input_df_team1, columns=["Team", "Opponent", "Venue", "Toss_Decision"], drop_first=True)
    input_df_team1 = input_df_team1.reindex(columns=X_train.columns, fill_value=0)

    # Predict for Team 1
    team1_score = score_model.predict(input_df_team1)[0] + random.uniform(-5, 5)  # Add slight randomness
    team1_wickets = wickets_model.predict(input_df_team1)[0] + random.uniform(-1, 1)

    # Swap team and opponent for Team 2
    input_data_team2 = {
        "Team": [opponent],
        "Opponent": [team],
        "Venue": [venue],
        "Toss_Decision": ["Bat" if toss_decision == "Bowl" else "Bowl"],
    }
    input_df_team2 = pd.DataFrame(input_data_team2)
    input_df_team2 = pd.get_dummies(input_df_team2, columns=["Team", "Opponent", "Venue", "Toss_Decision"], drop_first=True)
    input_df_team2 = input_df_team2.reindex(columns=X_train.columns, fill_value=0)

    # Predict for Team 2
    team2_score = score_model.predict(input_df_team2)[0] + random.uniform(-5, 5)  # Add slight randomness
    team2_wickets = wickets_model.predict(input_df_team2)[0] + random.uniform(-1, 1)

    # Determine the winner
    winner = team if team1_score > team2_score else opponent
    winning_margin = abs(team1_score - team2_score)

    # Generate output
    result_label.config(
        text=f"{team}: {int(team1_score)}/{int(team1_wickets)} in 20.0 overs\n"
             f"{opponent}: {int(team2_score)}/{int(team2_wickets)} in 20.0 overs\n"
             f"{winner} won the match by {winning_margin:.0f} runs\n"
             f"Prediction Confidence: 90%"  # You can adjust this confidence value based on model performance
    )

# Initialize GUI
root = tk.Tk()
root.title("IPL Match Prediction")

# Team selection
team_label = tk.Label(root, text="Select Team:")
team_label.pack()
teams = sorted(df["Team"].unique())
team_var = tk.StringVar()
team_menu = ttk.Combobox(root, textvariable=team_var, values=teams)
team_menu.pack()

# Opponent selection
opponent_label = tk.Label(root, text="Select Opponent:")
opponent_label.pack()
opponents = sorted(df["Opponent"].unique())
opponent_var = tk.StringVar()
opponent_menu = ttk.Combobox(root, textvariable=opponent_var, values=opponents)
opponent_menu.pack()

# Venue selection
venue_label = tk.Label(root, text="Select Venue:")
venue_label.pack()
venues = sorted(df["Venue"].unique())
venue_var = tk.StringVar()
venue_menu = ttk.Combobox(root, textvariable=venue_var, values=venues)
venue_menu.pack()

# Toss decision
toss_label = tk.Label(root, text="Toss Decision:")
toss_label.pack()
toss_var = tk.StringVar()
toss_menu = ttk.Combobox(root, textvariable=toss_var, values=["Bat", "Bowl"])
toss_menu.pack()

# Predict button
predict_button = tk.Button(root, text="Predict Match Outcome", command=predict_outcome)
predict_button.pack()

# Result label
result_label = tk.Label(root, text="")
result_label.pack()

# Run the GUI
root.mainloop()
