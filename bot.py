import os
import pickle
from dotenv import load_dotenv
import pandas as pd
from twitchio.ext import commands
import joblib

load_dotenv()


# Bot setup
class Bot(commands.Bot):
    def __init__(self):
        super().__init__(
            token=os.environ['TMI_TOKEN'],  # Your OAuth token
            client_id=os.environ['CLIENT_ID'],  # Your client ID
            nick=os.environ['BOT_NICK'],  # Your bot's username
            prefix='!',  # Command prefix
            initial_channels=[os.environ['CHANNEL']]  # Channels to join
        )
        # Add a whitelist of authorized users (lowercase usernames)
        self.whitelist = ['noidea100']  # Add your specific usernames here

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
    async def prediction_command(self, ctx, *, args=None):
        """
        Predict match outcome using format:
        !predict Team1, Team2, Map Name, Ban1, Ban2
        """
        # Check permissions - command only allowed for broadcaster, mods, or whitelisted users
        if not (ctx.author.is_broadcaster or ctx.author.is_mod or ctx.author.name.lower() in self.whitelist):
            return

        if not args:
            await ctx.send(
                "Format: !predict Team1, Team2, Map Name, Ban1, Ban2")
            return

        # Split the arguments by comma and strip whitespace
        parts = [part.strip() for part in args.split(',')]

        # Check if we have the correct number of arguments
        if len(parts) < 5:
            await ctx.send(
                "Not enough arguments! Format: !predict Team1, Team2, Map Name, Ban1, Ban2")
            return

        # Extract the values
        team_1 = parts[0]
        team_2 = parts[1]
        map_name = parts[2]
        ban_1 = parts[3]
        ban_2 = parts[4]

        # Optional: Get model name if provided as 6th argument
        model = "ensemble"  # Default model
        if len(parts) > 5:
            model = parts[5]

        # Now use these variables with your prediction model
        await ctx.send(f"Predicting match: {team_1} vs {team_2} on {map_name}")
        await ctx.send(f"Bans: {ban_1} / {ban_2} | Using model: {model}")

        # Prepare the input data
        input_data = pd.DataFrame({
            'team_name': [team_1],
            'team_name_opp': [team_2],
            'map_name': [map_name],
            'banned_hero': [ban_1],
            'banned_hero_opp': [ban_2]
        })

        try:
            # Preprocess the input data
            preprocessor = self.models['preprocessor']
            input_transformed = preprocessor.transform(input_data)

            # Get the model and make prediction
            selected_model = self.models[model]
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


# Run the bot
bot = Bot()
bot.run()