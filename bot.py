import discord
from discord.ext import commands
import json
import time
import asyncio
from llama_cpp import Llama
from collections import defaultdict
import logging
import re
import colorlog  # For colored logging

# Custom log format with optional username and user ID
LOG_FORMAT = "%(log_color)s[%(levelname)s] %(purple)s%(asctime)s%(reset)s - " \
             "%(light_red)s%(username)s%(yellow)s%(userid)s%(reset)s %(white)s%(message)s%(reset)s"
DATE_FORMAT = "%H:%M:%S"  # Minimal timestamp format

# Custom log colors
LOG_COLORS = {
    'DEBUG': 'cyan',
    'INFO': 'green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'red,bg_white',
}

# Secondary log colors for custom fields
SECONDARY_LOG_COLORS = {
    'username': {'light_red': 'light_red'},
    'userid': {'yellow': 'yellow'},
}

# Set up logging
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    LOG_FORMAT,
    datefmt=DATE_FORMAT,
    log_colors=LOG_COLORS,
    secondary_log_colors=SECONDARY_LOG_COLORS,
    reset=True,
    style='%'
))

logger = colorlog.getLogger('ModBot')
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Helper function to log with optional user info
def log_with_user_info(level: str, message: str, username: str = None, userid: str = None):
    """
    Logs a message with optional username and user ID.
    """
    extra = {
        'username': f"{username} " if username else "",
        'userid': f"({userid})" if userid else "",
    }
    logger.log(getattr(logging, level.upper()), message, extra=extra)

# Constants
BOT_SIGNATURE = "\n\n🐊 NEST RUSHERS AI - INFORMATION MAY NOT BE CORRECT"
RATE_LIMIT_SECONDS = 20
MAX_TOKENS = 200
MAX_CONTEXT_LENGTH = 450

# Initialize bot
intents = discord.Intents.default()
intents.message_content = True
bot = discord.Client(intents=intents)

# Rate limiting
user_last_message = defaultdict(float)

# Load knowledge base
try:
    with open('modding_knowledge.json', 'r') as f:
        knowledge_base = json.load(f)
    log_with_user_info("INFO", "Successfully loaded knowledge base")
except Exception as e:
    log_with_user_info("ERROR", f"Failed to load knowledge base: {e}")
    knowledge_base = {}

# Initialize Llama model
try:
    llm = Llama(
        model_path="C:/Users/User/Rusher AI/Models/llama-2-7b-chat.Q4_K_M.gguf",
        device="cuda:0",
        n_ctx=1536,  # 1536 is a sweet spot for 8GB VRAM
        n_batch=8    # Keep at 8 to avoid excessive VRAM spikes
    )
    log_with_user_info("INFO", "Successfully initialized Llama model")
except Exception as e:
    log_with_user_info("ERROR", f"Failed to initialize Llama model: {e}")
    raise

def find_relevant_knowledge(question: str) -> dict:
    """
    Optimized knowledge retrieval system using efficient matching and ranking.
    """
    log_with_user_info("INFO", f"Finding knowledge for question: {question}")
    relevant_knowledge = {}

    # Normalize question
    clean_question = question.lower().strip()
    words = set(clean_question.split())

    # Direct and content-based matching
    for category, content in knowledge_base.items():
        if isinstance(content, dict):
            for subcategory, info in content.items():
                subcategory_terms = set(subcategory.lower().split('_'))
                content_words = set(info.lower().split()) if isinstance(info, str) else set()

                # Prioritize exact subcategory match
                if words & subcategory_terms:
                    relevant_knowledge.setdefault(category, {})[subcategory] = info
                    continue  # Avoid duplicate checks

                # Check content for relevant words
                if words & content_words:
                    relevant_knowledge.setdefault(category, {})[subcategory] = info

        else:
            category_terms = set(category.lower().split('_'))
            if words & category_terms:
                relevant_knowledge[category] = content

    # Handle comparison-based queries more efficiently
    if {'difference', 'compare', 'vs', 'versus'} & words:
        for category, content in knowledge_base.items():
            if isinstance(content, dict):
                for subcategory in content.keys():
                    subcategory_terms = set(subcategory.lower().split('_'))
                    if words & subcategory_terms:
                        relevant_knowledge.setdefault(category, {})[subcategory] = content[subcategory]

    log_with_user_info("INFO", f"Relevant categories found: {list(relevant_knowledge.keys())}")
    return relevant_knowledge

def is_question(message: str) -> bool:
    """Enhanced question detection with better accuracy and handling of complex phrases."""
    message = message.lower().strip()
    
    # Early exit for simple cases with question mark
    if '?' in message:
        return True

    # List of question starting words (extended with variations like "how come")
    question_starts = [
        'how', 'what', 'where', 'when', 'why', 'can', 'could', 'would', 'is', 'are', 'does',
        'tell me', 'explain', 'do', 'did', 'does', 'will', 'should', 'which', 'who', 'whom',
        'how come', 'what if', 'why not', 'could you'
    ]
    
    # Check for short and unqualified queries like "how come"
    if message in ['how come', 'what if', 'why not']:  # Guard against short phrases
        return False
    
    # Match against common question starters, but with better context
    if any(message.startswith(word) for word in question_starts):
        # Check for the rest of the sentence; it's more likely a full question
        if len(message.split()) > 2:  # We want at least a subject + verb
            return True

    # Common request patterns (modding or gaming-related checks)
    modding_patterns = [
        'difference between', 'vs', 'versus', 'compare', 'tell me about', 'explain',
        'how to', 'how can', 'mod', 'modding', 'hack', 'game crash', 'cheat', 'unban', 'file editing'
    ]
    
    if any(pattern in message for pattern in modding_patterns):
        return True

    # Handle more complex phrasing with regular expressions
    question_regex = r'(\bhow\b.*\bcome\b|\bwhat\b.*\babout\b|\bwhy\b.*\bnot\b|\bwhere\b.*\bto\b|\bwho\b.*\belse\b)'
    if re.search(question_regex, message):
        return True
    
    return False

def is_rate_limited(user_id: int) -> bool:
    """Check if a user is rate limited."""
    current_time = time.time()
    if current_time - user_last_message[user_id] < RATE_LIMIT_SECONDS:
        return True
    user_last_message[user_id] = current_time
    return False

def summarize_knowledge(knowledge: dict) -> str:
    """Summarize the knowledge to fit within the context window."""
    summary = []
    for category, content in knowledge.items():
        if isinstance(content, dict):
            for subcategory, info in content.items():
                if isinstance(info, str):
                    summary.append(f"{subcategory}: {info[:100]}...")  # Truncate to 100 characters
        else:
            summary.append(f"{category}: {content[:100]}...")  # Truncate to 100 characters
    return "\n".join(summary)

async def generate_response(question: str) -> str | None:
    """Generate a response with improved knowledge retrieval."""
    try:
        log_with_user_info("INFO", f"Generating response for question: {question}")
        
        # Get relevant knowledge
        relevant_knowledge = find_relevant_knowledge(question)
        if not relevant_knowledge:
            log_with_user_info("INFO", "No relevant knowledge found")
            return None
            
        # Summarize the knowledge to fit within the context window
        knowledge_str = summarize_knowledge(relevant_knowledge)
        
        # Create focused prompt
        prompt = f"""You are a Dying Light modder. Based on this knowledge: {knowledge_str}
Question: {question}
Give a SIMPLE YET CONCISE answer (3-4 sentences max) and give a simple explanation/solution:"""

        log_with_user_info("INFO", f"Prompt length: {len(prompt)}")
        
        # response = llm(
        #     prompt,
        #     max_tokens=MAX_TOKENS,
        #     temperature=0.4,
        #     stop=["Question:", "\n\n"]
        # )
        response = await asyncio.to_thread(
            llm,
            prompt,
            max_tokens=MAX_TOKENS,
            temperature=0.3,
            stop=["Question:", "\n\n"],
            echo=False
        )
        
        answer = response['choices'][0]['text'].strip()
        return answer if answer.lower() != 'none' else None
        
    except Exception as e:
        log_with_user_info("ERROR", f"Error generating response: {e}")
        return None

@bot.event
async def on_ready():
    log_with_user_info("INFO", f"Bot {bot.user} is ready and online!")

@bot.event
async def on_message(message):
    if message.author.bot:
        return

    # Log user details
    log_with_user_info("INFO", f"Message: {message.content}", username=message.author.name, userid=str(message.author.id))

    if not is_question(message.content):
        return

    if is_rate_limited(message.author.id):
        log_with_user_info("WARNING", "User is rate limited", username=message.author.name, userid=str(message.author.id))
        return

    try:
        response = await generate_response(message.content)
        
        if response:
            # Create an embed
            embed = discord.Embed(
                description=response,
            )

            # Optionally, add fields or more customization:
            embed.set_footer(text=BOT_SIGNATURE)

            # Send the embed as a reply
            await message.reply(embed=embed)
    
    except discord.HTTPException as e:
        log_with_user_info("ERROR", f"HTTPException: {e}")
    except Exception as e:
        log_with_user_info("ERROR", f"Unexpected error: {e}")

# Run the bot
def run_bot(token: str):
    log_with_user_info("INFO", "Starting bot...")
    bot.run(token)

run_bot("YOUR_TOKEN_HERE")