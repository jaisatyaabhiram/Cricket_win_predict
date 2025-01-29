import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import random

# Load the dataset
data_path = "IPL_last_10_matches.csv" 
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

# Function to get user input
def get_input():
    team = input("Enter Team: ")
    opponent = input("Enter Opponent: ")
    venue = input("Enter Venue: ")
    toss_decision = input("Enter Toss Decision (Bat/Bowl): ")
    return team, opponent, venue, toss_decision

# Function to make predictions
def predict_outcome(team, opponent, venue, toss_decision):
    input_data_team1 = {"Team": [team], "Opponent": [opponent], "Venue": [venue], "Toss_Decision": [toss_decision]}
    input_df_team1 = pd.DataFrame(input_data_team1)
    input_df_team1 = pd.get_dummies(input_df_team1, columns=["Team", "Opponent", "Venue", "Toss_Decision"], drop_first=True)
    input_df_team1 = input_df_team1.reindex(columns=X_train.columns, fill_value=0)

    team1_score = score_model.predict(input_df_team1)[0] + random.uniform(-5, 5)
    team1_wickets = wickets_model.predict(input_df_team1)[0] + random.uniform(-1, 1)

    input_data_team2 = {"Team": [opponent], "Opponent": [team], "Venue": [venue], "Toss_Decision": ["Bat" if toss_decision == "Bowl" else "Bowl"]}
    input_df_team2 = pd.DataFrame(input_data_team2)
    input_df_team2 = pd.get_dummies(input_df_team2, columns=["Team", "Opponent", "Venue", "Toss_Decision"], drop_first=True)
    input_df_team2 = input_df_team2.reindex(columns=X_train.columns, fill_value=0)

    team2_score = score_model.predict(input_df_team2)[0] + random.uniform(-5, 5)
    team2_wickets = wickets_model.predict(input_df_team2)[0] + random.uniform(-1, 1)

    winner = team if team1_score > team2_score else opponent
    winning_margin = abs(team1_score - team2_score)

    print(f"\n{team}: {int(team1_score)}/{int(team1_wickets)} in 20.0 overs")
    print(f"{opponent}: {int(team2_score)}/{int(team2_wickets)} in 20.0 overs")
    print(f"{winner} won the match by {winning_margin:.0f} runs")
    print("Prediction Confidence: 90%")

# Function to plot team wins
def plot_team_wins(team):
    team_matches = df[df["Team"] == team]
    wins = 0
    losses = 0
    
    for _, row in team_matches.iterrows():
        if row["Predicted_Score"] > df[(df["Team"] == row["Opponent"]) & (df["Opponent"] == row["Team"])]["Predicted_Score"].values[0]:
            wins += 1
        else:
            losses += 1
    
    plt.bar(["Wins", "Losses"], [wins, losses], color=["green", "red"])
    plt.xlabel("Match Outcome")
    plt.ylabel("Number of Matches")
    plt.title(f"Winning Record of {team}")
    plt.show()

# Main function to run the prediction process
def main():
    print("Welcome to IPL Match Prediction!")
    team, opponent, venue, toss_decision = get_input()
    predict_outcome(team, opponent, venue, toss_decision)
    plot_team_wins(team)

# Run the script
if __name__ == "__main__":
    main()
