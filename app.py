import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load and clean dataset
@st.cache_data
def load_data():
    df = pd.read_csv("ipl_players.csv")
    df['Role'] = df['Role'].str.lower()
    df['Team'] = df['Team'].str.upper()
    df['Name_clean'] = df['Name'].str.strip().str.lower()
    return df

# Core team members
core_teams = {
    "MI": ["Jasprit Bumrah", "Suryakumar Yadav", "Hardik Pandya", "Rohit Sharma", "Tilak Varma"],
    "CSK": ["Ruturaj Gaikwad", "Matheesha Pathirana", "Shivam Dube", "Ravindra Jadeja", "MS Dhoni"],
    "SRH": ["Pat Cummins", "Abhishek Sharma", "Ishan Kishan", "Mohammed Shami"],
    "DC": ["Axar Patel", "Kuldeep Yadav", "KL Rahul", "Mitchell Starc"],
    "GT": ["Rashid Khan", "Shubman Gill", "Jos Buttler", "Kagiso Rabada"],
    "LSG": ["Rishabh Pant", "David Miller", "Avesh Khan"],
    "RCB": ["Virat Kohli", "AB devillers", "Chris Gayle"],
    "KKR": ["Sunil Narine"],
    "RR": ["Yashasvi Jaiswal"],
    "PBKS": ["KL Rahul", "Arshdeep Singh"]
}

core_player_to_team = {
    name.strip().lower(): team for team, names in core_teams.items() for name in names
}

def get_top_similar(df, role, team_name, top_n):
    role_players = df[df['Role'] == role].copy()

    features = ['Batting_Average', 'Strike_Rate', 'Bowling_Average', 'Matches']
    scaler = StandardScaler()
    role_players_scaled = scaler.fit_transform(role_players[features].fillna(0))

    ideal_player = role_players[role_players['Team'] == team_name].mean(numeric_only=True)
    ideal_vector = scaler.transform([ideal_player[features].fillna(0)])

    similarities = cosine_similarity(role_players_scaled, ideal_vector).flatten()
    role_players['Similarity'] = similarities

    # Filter core player conflicts
    def is_valid(row):
        name = row['Name_clean']
        if name in core_player_to_team:
            return core_player_to_team[name] == team_name
        return True

    role_players = role_players[role_players.apply(is_valid, axis=1)]
    return role_players.sort_values(by='Similarity', ascending=False).head(top_n)

def local_css():
    st.markdown("""
    <style>
    html, body, .main {
        background-color: #2c5a99 !important;
        color: #f2f2f2 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        padding: 2rem;
    }
    .block-container {
        max-width: 800px;
        margin: auto;
    }
    .app-heading {
        font-size: 4rem;
        font-weight: bold;
        color: #f9a825;
        text-align: center;
        text-shadow: 2px 2px 5px #000;
        margin-bottom: 2rem;
    }
    .stTextInput > div > div > input,
    .stSelectbox > div > div > div > select {
        background-color: #14325c !important;
        color: white !important;
        border: 2px solid #f9a825 !important;
        padding: 0.6rem;
        border-radius: 10px;
        font-weight: 600;
    }
    div.stButton > button:first-child {
        background-color: #f9a825 !important;
        color: #0a1d37 !important;
        font-weight: bold;
        font-size: 1.2rem;
        padding: 0.6rem 1.5rem;
        border-radius: 25px;
        box-shadow: 0 5px 15px rgba(249, 168, 37, 0.5);
        margin-top: 1rem;
    }
    div.stButton > button:first-child:hover {
        background-color: #ffcc33 !important;
        color: #0a1d37 !important;
    }
    .player-card {
        background: linear-gradient(145deg, #123962, #0f2c4a);
        padding: 20px;
        margin-bottom: 1.2rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        color: #fff;
    }
    .player-name {
        font-size: 1.6rem;
        font-weight: bold;
        color: #f9a825;
        text-shadow: 1px 1px 3px #000;
        margin-bottom: 0.5rem;
    }
    .footer {
        text-align: center;
        color: #888;
        margin-top: 2rem;
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)

def recommend_players(team_name, missing_roles, df, top_n=5):
    team_name = team_name.strip().upper()
    roles = [role.strip().lower() for role in missing_roles.split(',')]
    st.markdown("<div class='app-heading'>Recommended Players</div>", unsafe_allow_html=True)

    for role in roles:
        top_players = get_top_similar(df, role, team_name, top_n)
        if top_players.empty:
            st.warning(f"No players available for role: {role}")
        else:
            st.markdown(f"### Top {top_n} {role.capitalize()}s:")
            for _, row in top_players.iterrows():
                st.markdown(f"""
                <div class='player-card'>
                    <div class='player-name'>{row['Name']}</div>
                    <div><b>Team:</b> {row['Team']} | <b>Bat Avg:</b> {row['Batting_Average']} | 
                    <b>Strike Rate:</b> {row['Strike_Rate']} | <b>Bowl Avg:</b> {row['Bowling_Average']}</div>
                </div>""", unsafe_allow_html=True)

def main():
    local_css()
    st.markdown("<div class='app-heading'>IPL Player Recommendation System</div>", unsafe_allow_html=True)
    df = load_data()
    team_input = st.selectbox("Select IPL Team:", sorted(df['Team'].unique()))
    roles_input = st.text_input("Enter missing roles (comma separated, e.g., bowler, all-rounder):", value="bowler, all-rounder")

    if st.button("Recommend Players"):
        if not roles_input.strip():
            st.error("Please enter at least one role.")
        else:
            recommend_players(team_input, roles_input, df)

    st.markdown("<div class='footer'>Developed by Vaish | Powered by Streamlit</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
