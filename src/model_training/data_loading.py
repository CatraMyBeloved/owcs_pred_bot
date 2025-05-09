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

import sqlite3
import pandas as pd
import numpy as np

table_names = ["teams", "bans", "hero_composition", "heroes",
               "maps", "match_maps", "matches", "rounds"]

def load_data_from_sqlite(table_name: str,db_path: str = "../../data/owcs.db") \
        -> (
        pd.DataFrame):
    """
    Load data from a SQLite database into a Pandas DataFrame.

    Args:
        db_path (str): Path to the SQLite database file.
        table_name (str): Name of the table to load.

    Returns:
        pd.DataFrame: DataFrame containing the loaded data.
    """
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)

    # Load data from the specified table into a DataFrame
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

    # Close the connection
    conn.close()

    return df

def load_all_tables(db_path: str = "../../data/owcs.db") -> dict:
    """
    Load all tables from the SQLite database into a dictionary of DataFrames.

    Args:
        db_path (str): Path to the SQLite database file.

    Returns:
        dict: Dictionary containing DataFrames for each table.
    """
    data = {}
    for table_name in table_names:
        data[table_name] = load_data_from_sqlite(table_name, db_path)
    return data

data = load_all_tables()

hero_composition = data["hero_composition"]
rounds = data["rounds"]
match_maps = data["match_maps"]
matches = data["matches"]
teams = data["teams"]
heroes = data["heroes"]
maps = data["maps"]

def determine_iswin(row: pd.Series) -> int:
    if row["team"] == row["map_win_team_id"]:
        return 1
    else:
        return 0

def join_all_tables() -> pd.DataFrame:
    """
    Join all tables in the database to create a comprehensive DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing the joined data.
    """
    # Join the tables using the appropriate keys
    df = pd.merge(hero_composition, heroes, on = "hero_id")
    df = pd.merge(df, rounds, on="round_id")
    df = pd.merge(df, match_maps, on="match_map_id")
    df = pd.merge(df, matches, on="match_id")
    df = pd.merge(df, teams, left_on="team", right_on="team_id")
    df = pd.merge(df, maps, on="map_id")
    df["is_win"]  = df.apply(determine_iswin, axis=1)

    return df

test = join_all_tables()

print(test.columns)