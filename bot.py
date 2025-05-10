import asyncio
import logging
import sqlite3
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os

import asqlite
import twitchio
from twitchio.ext import commands
from twitchio import eventsub
from dotenv import load_dotenv

load_dotenv()


LOGGER: logging.Logger = logging.getLogger("Bot")

CLIENT_ID: str = os.getenv("TWITCH_CLIENT_ID")
CLIENT_SECRET: str = os.getenv("TWITCH_CLIENT_SECRET")
BOT_ID = os.getenv("TWITCH_BOT_ID")
OWNER_ID = os.getenv("TWITCH_OWNER_ID")



class Bot(commands.Bot):
    def __init__(self, *, token_database: asqlite.Pool) -> None:
        self.token_database = token_database
        super().__init__(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            bot_id=BOT_ID,
            owner_id=OWNER_ID,
            prefix="!",
        )
        self.model_load_dict = {
            "random_forest" : "./models/random_forest.pkl",
            "extra_trees" : "./models/extra_trees.pkl",
            "neural_network" : "./models/neural_network.pkl",
            "ensemble" : "./models/ensemble.pkl",
            "preprocessor" : "./models/preprocessor.pkl"
        }
        self.models = {}

        for model in self.model_load_dict.keys():
            model_path = self.model_load_dict[model]
            if os.path.exists(model_path):
                self.models[model] = joblib.load(model_path)
                print(f"Loaded model {model} from {model_path}")
            else:
                print(f"Model {model} not found at {model_path}")


    async def setup_hook(self) -> None:
         # Add our component which contains our commands...
         await self.add_component(MyComponent(self))

         # Subscribe to read chat (event_message) from our channel as the bot...
         # This creates and opens a websocket to Twitch EventSub...
         subscription = eventsub.ChatMessageSubscription(broadcaster_user_id=OWNER_ID, user_id=BOT_ID)
         await self.subscribe_websocket(payload=subscription)

         # Subscribe and listen to when a stream goes live
         # For this example listen to our own stream...
         subscription = eventsub.StreamOnlineSubscription(broadcaster_user_id=OWNER_ID)
         await self.subscribe_websocket(payload=subscription)

    async def add_token(self, token: str, refresh: str) -> twitchio.authentication.ValidateTokenPayload:
        # Make sure to call super() as it will add the tokens interally and return us some data...
        resp: twitchio.authentication.ValidateTokenPayload = await super().add_token(token, refresh)

        # Store our tokens in a simple SQLite Database when they are authorized...
        query = """
        INSERT INTO tokens (user_id, token, refresh)
        VALUES (?, ?, ?)
        ON CONFLICT(user_id)
        DO UPDATE SET
            token = excluded.token,
            refresh = excluded.refresh;
        """

        async with self.token_database.acquire() as connection:
            await connection.execute(query, (resp.user_id, token, refresh))

        LOGGER.info("Added token to the database for user: %s", resp.user_id)
        return resp

    async def load_tokens(self, path: str | None = None) -> None:
        # We don't need to call this manually, it is called in .login() from .start() internally...

        async with self.token_database.acquire() as connection:
            rows: list[sqlite3.Row] = await connection.fetchall("""SELECT * from tokens""")

        for row in rows:
            await self.add_token(row["token"], row["refresh"])

    async def setup_database(self) -> None:
        # Create our token table, if it doesn't exist
        query = """CREATE TABLE IF NOT EXISTS tokens(user_id TEXT PRIMARY KEY, token TEXT NOT NULL, refresh TEXT NOT NULL)"""
        async with self.token_database.acquire() as connection:
            await connection.execute(query)

    async def event_ready(self) -> None:
        LOGGER.info("Successfully logged in as: %s", self.bot_id)


class MyComponent(commands.Component):
    def __init__(self, bot: Bot):
        # Passing args is not required...
        # We pass bot here as an example...
        self.bot = bot

    # We use a listener in our Component to display the messages received.
    @commands.Component.listener()
    async def event_message(self, payload: twitchio.ChatMessage) -> None:
        print(f"[{payload.broadcaster.name}] - {payload.chatter.name}: {payload.text}")

    @commands.command(aliases=["hello", "howdy", "hey"])
    async def hi(self, ctx: commands.Context) -> None:
        """Simple command that says hello!

        !hi, !hello, !howdy, !hey
        """
        await ctx.reply(f"Hello {ctx.chatter.mention}!")

    @commands.command(aliases=["modellist"])
    async def models(self, ctx: commands.Context) -> None:
        """List all available models"""
        model_list = "\n".join(self.bot.models.keys())
        await ctx.send(f"Available models:\n{model_list}")

    @commands.command()
    @commands.is_elevated()
    async def predict(self, ctx: commands.Context, *, content: str) -> None:
        arguments = [argument.strip() for argument in content.split(",")]

        model = "ensemble"

        if len(arguments) not in [5, 6]:
            await ctx.send("Format error. Format: predict team_1, team_2, "
                           "map, ban_team_1, ban_team_2, model(optional)")
            return

        if len(arguments) == 6:
            team_1, team_2, map_name, ban_1, ban_2, model = arguments
        else:
            team_1, team_2, map_name, ban_1, ban_2 = arguments

        input_data = pd.DataFrame({
            'team_name': [team_1],
            'team_name_opp': [team_2],
            'map_name': [map_name],
            'banned_hero': [ban_1],
            'banned_hero_opp': [ban_2]
        })

        try:
            # Preprocess the input data
            preprocessor = self.bot.models['preprocessor']
            input_transformed = preprocessor.transform(input_data)

            # Get the model and make prediction
            selected_model = self.bot.models[model]
            prediction_prob = \
                selected_model.predict_proba(input_transformed)[0][1]

            # Determine winner and confidence
            winner = team_1 if prediction_prob > 0.5 else team_2
            confidence = max(prediction_prob, 1 - prediction_prob)

            # Format the prediction message
            message = f"Prediction using {model} model:\n"
            message += f"• {winner} is predicted to win ({confidence:.2%} confidence)\n"
            message += f"• Map: {map_name} | Bans: {ban_1} & {ban_2}"

            await ctx.send(message)
        except Exception as e:
            await ctx.send(f"Error making prediction: {str(e)}")


    @commands.Component.listener()
    async def event_stream_online(self, payload: twitchio.StreamOnline) -> None:
        # Event dispatched when a user goes live from the subscription we made above...

        # Keep in mind we are assuming this is for ourselves
        # others may not want your bot randomly sending messages...
        await payload.broadcaster.send_message(
            sender=self.bot.bot_id,
            message=f"Hi... {payload.broadcaster}! You are live!",
        )


def main() -> None:
    twitchio.utils.setup_logging(level=logging.INFO)

    async def runner() -> None:
        async with asqlite.create_pool("tokens.db") as tdb, Bot(token_database=tdb) as bot:
            await bot.setup_database()
            await bot.start()

    try:
        asyncio.run(runner())
    except KeyboardInterrupt:
        LOGGER.warning("Shutting down due to KeyboardInterrupt...")


if __name__ == "__main__":
    main()