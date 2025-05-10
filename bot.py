import os
import pickle

import pandas as pd
from twitchio.ext import commands
import joblib


# Bot setup
class Bot(commands.Bot):
    def __init__(self):
        # Replace with your own credentials
        super().__init__(
            token='YOUR_ACCESS_TOKEN',  # Your OAuth token
            client_id='YOUR_CLIENT_ID',  # Your client ID
            nick='YOUR_BOT_USERNAME',  # Your bot's username
            prefix='pred:',  # Command prefix
            initial_channels=['YOUR_CHANNEL']  # Channels to join
        )
        # load models from models directory
        self.model_load_dict = {
            'preprocessor': './models/preprocessor.pkl',
            'rand_forest': './models/random_forest.pkl',
            'nn': './models/neural_network.pkl',
            'extra_trees': './models/extra_trees.pkl',
            'ensemble': './models/ensemble.pkl',
        }

        self.models = {}

        # Load models
        for model in self.model_load_dict.keys():
            model_path = self.model_load_dict[model]
            if os.path.exists(model_path):
                self.models[model] = joblib.load(model_path)
                print(f"Loaded model {model} from {model_path}")
            else:
                print(f"Model {model} not found at {model_path}")

    async def event_ready(self):
        """Called once when the bot goes online."""
        print(f"Bot is online! | {self.nick}")
        print(f"User ID: {self.user_id}")

    async def event_message(self, message):
        """Called when a message is sent in chat."""
        # Ignore messages from the bot itself
        if message.echo:
            return

        # Handle command processing
        await self.handle_commands(message)

    @commands.command(name='predict')
    async def prediction_command(self, ctx, team_1: str, team_2: str, map: str,
                            ban_1: str, ban_2: str, model: str = 'ensemble'):

        if not all([team_1, team_2, map, ban_1, ban_2]):
            await ctx.send("Please provide all required arguments.")
            return

        # Check if the model is loaded
        if model not in self.models:
            await ctx.send(f"Model {model} is not loaded.")
            return

        # Prepare the input data
        input_data = pd.DataFrame({
            'team_name': team_1,
            'team_name_opp': team_2,
            'map_name': map,
            'banned_hero': ban_1,
            'banned_hero_opp': ban_2
        })

        # Preprocess the input data
        preprocessor = self.models['preprocessor']
        input_transformed = preprocessor.transform(input_data)
        # Get the model
        model = self.models[model]

        await ctx.send(f'Hello {ctx.author.name}!')


# Run the bot
bot = Bot()
bot.run()