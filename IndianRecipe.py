"""
Streamlit Conversational AI Recipe Chatbot
"""

import json
import os
import re
import streamlit as st
from datetime import datetime
from typing import Dict, List, Optional
from collections import Counter

# Try to import OpenAI with proper error handling
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

class ConversationalRecipeBot:
    """A naturally conversational AI cooking assistant with deep Indian cuisine knowledge"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.recipes_file = "recipes.json"
        self.chat_history_file = "chat_history.json"
        self.preferences_file = "cooking_preferences.json"
        
        self.recipes = self.load_recipes()
        self.cooking_preferences = self.load_preferences()
        
        # Initialize OpenAI client properly
        self.openai_client = None
        if api_key and OPENAI_AVAILABLE:
            try:
                self.openai_client = OpenAI(api_key=api_key)
                # Test the connection
                test_response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=5
                )
                st.success("✅ OpenAI API connected successfully!")
            except Exception as e:
                st.error(f"⚠️  OpenAI API connection failed: {e}")
                self.openai_client = None
        else:
            if not api_key:
                st.warning("⚠️  No OpenAI API key provided. Using basic responses.")
            else:
                st.error("⚠️  OpenAI library not installed. Run: pip install openai")
        
        if len(self.recipes) > 0:
            self.analyze_cooking_style()
    
    def load_recipes(self) -> Dict:
        """Load recipes from file"""
        if os.path.exists(self.recipes_file):
            try:
                with open(self.recipes_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                st.error(f"Error loading recipes: {e}")
                return {}
        return {}
    
    def load_preferences(self) -> Dict:
        """Load learned cooking preferences"""
        if os.path.exists(self.preferences_file):
            try:
                with open(self.preferences_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                st.error(f"Error loading preferences: {e}")
                return {}
        return {}
    
    def save_recipes(self):
        """Save recipes to file"""
        try:
            with open(self.recipes_file, 'w', encoding='utf-8') as f:
                json.dump(self.recipes, f, indent=2, ensure_ascii=False)
        except Exception as e:
            st.error(f"Error saving recipes: {e}")
    
    def save_preferences(self):
        """Save learned preferences"""
        try:
            with open(self.preferences_file, 'w', encoding='utf-8') as f:
                json.dump(self.cooking_preferences, f, indent=2)
        except Exception as e:
            st.error(f"Error saving preferences: {e}")

    def analyze_cooking_style(self):
        """Dynamically analyze user's cooking patterns and preferences"""
        if not self.recipes:
            return
        
        analysis = {
            'favorite_cuisines': Counter(),
            'spice_preferences': Counter(),
            'cooking_methods': Counter(),
            'common_ingredients': Counter(),
            'dietary_patterns': Counter(),
            'meal_complexity': Counter(),
            'flavor_profiles': [],
            'cooking_frequency_indicators': [],
            'special_techniques': set(),
            'regional_specialties': Counter()
        }
        
        for recipe_key, recipe in self.recipes.items():
            # Cuisine analysis
            region = recipe.get('region', 'unknown').lower()
            analysis['favorite_cuisines'][region] += 1
            
            # Spice level analysis
            spice_level = recipe.get('spice_level', 'medium').lower()
            analysis['spice_preferences'][spice_level] += 1
            
            # Difficulty analysis
            difficulty = recipe.get('difficulty', 'medium').lower()
            analysis['meal_complexity'][difficulty] += 1
            
            # Dietary patterns
            dietary = recipe.get('dietary', [])
            for diet in dietary:
                analysis['dietary_patterns'][diet.lower()] += 1
            
            # Ingredients analysis
            ingredients = recipe.get('ingredients', [])
            for ingredient in ingredients:
                ingredient_lower = ingredient.lower()
                analysis['common_ingredients'][ingredient_lower] += 1
        
        # Update preferences
        self.cooking_preferences = {
            'analysis_date': datetime.now().isoformat(),
            'total_recipes': len(self.recipes),
            'patterns': {k: dict(v.most_common(10)) if hasattr(v, 'most_common') else v 
                        for k, v in analysis.items()},
            'personality_insights': self._generate_personality_insights(analysis)
        }
        
        self.save_preferences()
    
    def _generate_personality_insights(self, analysis) -> Dict:
        """Generate cooking personality insights from analysis"""
        insights = {}
        
        # Cuisine preference
        top_cuisine = analysis['favorite_cuisines'].most_common(1)
        if top_cuisine:
            insights['cuisine_love'] = top_cuisine[0][0]
        
        # Spice tolerance
        spice_prefs = analysis['spice_preferences']
        if spice_prefs.get('high', 0) > spice_prefs.get('mild', 0):
            insights['spice_lover'] = True
        else:
            insights['spice_lover'] = False
        
        # Cooking complexity preference
        complexity_prefs = analysis['meal_complexity']
        if complexity_prefs.get('hard', 0) > complexity_prefs.get('easy', 0):
            insights['complexity_seeker'] = True
        else:
            insights['complexity_seeker'] = False
        
        # Regional expertise
        regional = analysis['regional_specialties']
        if regional:
            insights['regional_expertise'] = regional.most_common(1)[0][0]
        
        # Special skills
        insights['special_skills'] = list(analysis['special_techniques'])
        
        return insights

    def chat(self, user_message: str) -> str:
        """Main chat function - naturally conversational responses"""
        # Generate response
        if self.openai_client:
            try:
                response = self._get_ai_response(user_message)
            except Exception as e:
                st.error(f"AI response failed: {e}")
                response = self._get_fallback_response(user_message)
        else:
            response = self._get_fallback_response(user_message)
        
        return response
    
    def _get_ai_response(self, user_message: str) -> str:
        """Generate natural AI response using learned knowledge"""
        if not self.openai_client:
            raise Exception("No OpenAI client available")
            
        # Build dynamic context
        learned_context = self._build_dynamic_context()
        
        # Recent conversation context from Streamlit session state
        recent_context = ""
        if "messages" in st.session_state and len(st.session_state.messages) > 1:
            recent_exchanges = st.session_state.messages[-10:]  # Last 5 pairs
            for i in range(0, len(recent_exchanges) - 1, 2):
                if i + 1 < len(recent_exchanges):
                    recent_context += f"Human: {recent_exchanges[i]['content']}\nAssistant: {recent_exchanges[i+1]['content']}\n\n"
        
        system_prompt = f"""You are an extremely knowledgeable and conversational cooking assistant who specializes in Indian cuisine. You chat naturally like a passionate cooking friend who has learned from the user's personal recipe collection.

{learned_context}

CONVERSATION STYLE:
- Chat naturally and conversationally, like talking to a cooking-passionate friend
- Be enthusiastic and knowledgeable about cooking, especially Indian cuisine
- Reference specific recipes, ingredients, or techniques from what you've learned when relevant
- Offer practical tips, variations, and cooking wisdom
- Ask engaging follow-up questions to keep the conversation flowing
- Share interesting food facts, cultural context, or cooking stories when appropriate
- Be warm, friendly, and genuinely helpful

INDIAN CUISINE EXPERTISE:
- Deep knowledge of regional Indian cuisines, spices, techniques, and traditions
- Understanding of spice combinations, tempering (tadka), pressure cooking, fermentation
- Knowledge of ingredient substitutions, cooking ratios, timing techniques
- Familiarity with dietary patterns (vegetarian, vegan, Jain, regional variations)
- Cultural context around Indian food traditions and meal structures

RECENT CONVERSATION:
{recent_context}

Respond naturally as a knowledgeable cooking friend who has learned from their recipes and loves talking about food, especially Indian cuisine."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=600,
                temperature=0.8,
                top_p=0.9
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise e
    
    def _build_dynamic_context(self) -> str:
        """Build dynamic context from learned recipes and preferences"""
        if not self.recipes:
            return "I'm ready to learn from your recipes! Add some and I'll understand your cooking style better."
        
        context = f"LEARNED KNOWLEDGE FROM {len(self.recipes)} RECIPES:\n\n"
        
        # Add some basic context
        recent_recipes = list(self.recipes.items())[-3:]
        context += f"📋 RECENT RECIPES I KNOW:\n"
        for recipe_key, recipe in recent_recipes:
            name = recipe.get('name', recipe_key)
            region = recipe.get('region', 'Unknown region')
            context += f"- {name} ({region})\n"
        
        context += f"\nI can discuss any of these recipes, suggest variations, help with cooking techniques, or chat about anything food-related!"
        return context
    
    def _get_fallback_response(self, user_message: str) -> str:
        """Enhanced fallback response when AI is not available"""
        message_lower = user_message.lower().strip()
        
        # Handle empty messages
        if not message_lower:
            return "I'm here and ready to chat! What would you like to talk about?"
        
        # Greeting responses
        if any(word in message_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good evening']):
            cooking_context = ""
            if self.recipes:
                cooking_context = f" I've learned {len(self.recipes)} of your recipes, so I'm great with cooking questions,"
            return f"Hello!{cooking_context} but I'm happy to chat about anything. What's on your mind?"
        
        # Cooking-related queries
        if any(word in message_lower for word in ['recipe', 'cook', 'make', 'prepare', 'dish', 'spice', 'ingredient', 'food', 'kitchen']):
            if self.recipes:
                recipe_names = [recipe.get('name', 'Unnamed recipe') for recipe in list(self.recipes.values())[:3]]
                return f"Great! I can help with cooking. I know your recipes like: {', '.join(recipe_names)}. What would you like to know?"
            else:
                return "I'd love to help with cooking! I specialize in recipes and cooking techniques. What can I help you with?"
        
        # Indian cuisine specific
        if any(word in message_lower for word in ['indian', 'curry', 'masala', 'tadka', 'dal', 'roti', 'naan']):
            return "I absolutely love Indian cuisine! What would you like to know - specific dishes, techniques, regional specialties, or spice combinations?"
        
        # Default responses
        import random
        responses = [
            "I'm not sure I fully understand, but I'm here to help! Could you tell me more about what you're looking for?",
            "That's interesting! While cooking is my specialty, I'm happy to chat about other topics. Can you elaborate?",
            f"I'd love to help! I know the most about cooking (I've learned {len(self.recipes)} of your recipes), but feel free to ask me anything.",
            "I'm here to chat! What would you like to talk about? Cooking questions are my forte, but I'm open to other topics too."
        ]
        
        return random.choice(responses)

    def quick_add_recipe(self, name: str, ingredients: List[str], instructions: List[str], 
                        region: str = None, **kwargs) -> bool:
        """Quick way to add a recipe"""
        recipe = {
            "name": name,
            "ingredients": ingredients,
            "instructions": instructions,
            "difficulty": kwargs.get('difficulty', 'medium'),
            "dietary": kwargs.get('dietary', ['vegetarian']),
            "added_at": datetime.now().isoformat(),
            **kwargs
        }
        
        if region:
            recipe["region"] = region
            
        # Generate key
        key = re.sub(r'[^a-z0-9_]', '', name.lower().replace(' ', '_'))
        
        try:
            self.recipes[key] = recipe
            self.save_recipes()
            self.analyze_cooking_style()
            return True
        except Exception as e:
            st.error(f"Error adding recipe: {e}")
            return False

    def get_cooking_insights(self) -> str:
        """Get insights about learned cooking patterns"""
        if not self.cooking_preferences:
            return "Add some recipes and I'll analyze your cooking style!"
        
        insights = self.cooking_preferences.get('personality_insights', {})
        
        response = f"🔍 **Your Cooking Style Analysis** (from {len(self.recipes)} recipes):\n\n"
        
        if insights.get('cuisine_love'):
            response += f"🍛 **Cuisine Love**: You're really into {insights['cuisine_love']} food!\n"
        
        if insights.get('regional_expertise'):
            response += f"🇮🇳 **Regional Specialty**: You've mastered {insights['regional_expertise'].replace('_', ' ')} cuisine\n"
        
        if insights.get('spice_lover'):
            response += f"🌶️ **Spice Profile**: You love bold, spicy flavors!\n"
        else:
            response += f"🌿 **Spice Profile**: You prefer mild to medium spice levels\n"
        
        skills = insights.get('special_skills', [])
        if skills:
            response += f"⭐ **Special Skills**: {', '.join([skill.replace('_', ' ') for skill in skills])}\n"
        
        return response


def main():
    """Main Streamlit app"""
    st.set_page_config(page_title="Recipe AI Chatbot", page_icon="🍛", layout="wide")
    
    st.title("🍛 Conversational Recipe AI")
    st.markdown("*I learn from your recipes and chat naturally about cooking!*")
    
    # Initialize session state
    if "bot" not in st.session_state:
        st.session_state.bot = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "api_key_set" not in st.session_state:
        st.session_state.api_key_set = False
    
    # Sidebar for configuration and features
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Key input
        if not st.session_state.api_key_set:
            api_key = st.text_input("OpenAI API Key (optional)", type="password", 
                                   help="For enhanced AI responses. Leave empty for basic mode.")
            
            if st.button("Initialize Bot"):
                st.session_state.bot = ConversationalRecipeBot(api_key if api_key else None)
                st.session_state.api_key_set = True
                
                # Add sample recipe if none exist
                if len(st.session_state.bot.recipes) == 0:
                    st.session_state.bot.quick_add_recipe(
                        name="Simple Dal Tadka",
                        ingredients=[
                            "1 cup yellow dal (moong or toor)",
                            "2 cups water",
                            "1/2 tsp turmeric",
                            "1 tsp salt",
                            "1 tbsp ghee",
                            "1 tsp cumin seeds",
                            "2 dried red chilies",
                            "1 onion, chopped",
                            "2 tomatoes, chopped",
                            "1 tsp ginger-garlic paste"
                        ],
                        instructions=[
                            "Wash dal and cook with water, turmeric, and salt until soft",
                            "Heat ghee in pan, add cumin seeds and red chilies",
                            "When cumin splutters, add onions and cook until golden",
                            "Add ginger-garlic paste, cook for 1 minute",
                            "Add tomatoes, cook until mushy",
                            "Pour this tempering over cooked dal",
                            "Simmer for 5 minutes, garnish with cilantro"
                        ],
                        region="Indian",
                        spice_level="mild",
                        difficulty="easy",
                        dietary=["vegetarian"]
                    )
                    st.success("✅ Added a sample recipe to get started!")
                st.rerun()
        
        if st.session_state.bot:
            st.success(f"🤖 Bot initialized with {len(st.session_state.bot.recipes)} recipes")
            
            st.header("📊 Quick Actions")
            
            if st.button("Show My Cooking Style"):
                insights = st.session_state.bot.get_cooking_insights()
                st.markdown(insights)
            
            if st.button("Clear Chat History"):
                st.session_state.messages = []
                st.rerun()
            
            # Recipe management
            st.header("🍳 Recipe Management")
            
            with st.expander("Add New Recipe"):
                with st.form("add_recipe"):
                    recipe_name = st.text_input("Recipe Name")
                    recipe_region = st.selectbox("Region/Cuisine", 
                                               ["Indian", "Chinese", "Italian", "Mexican", "American", "Other"])
                    recipe_difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard"])
                    recipe_spice = st.selectbox("Spice Level", ["mild", "medium", "high"])
                    
                    ingredients_text = st.text_area("Ingredients (one per line)")
                    instructions_text = st.text_area("Instructions (one per line)")
                    
                    if st.form_submit_button("Add Recipe"):
                        if recipe_name and ingredients_text and instructions_text:
                            ingredients = [ing.strip() for ing in ingredients_text.split('\n') if ing.strip()]
                            instructions = [inst.strip() for inst in instructions_text.split('\n') if inst.strip()]
                            
                            if st.session_state.bot.quick_add_recipe(
                                name=recipe_name,
                                ingredients=ingredients,
                                instructions=instructions,
                                region=recipe_region,
                                difficulty=recipe_difficulty,
                                spice_level=recipe_spice,
                                dietary=["vegetarian"]  # Default
                            ):
                                st.success(f"✅ Added recipe: {recipe_name}")
                                st.rerun()
                        else:
                            st.error("Please fill in all fields")
            
            # Show existing recipes
            if st.session_state.bot.recipes:
                with st.expander(f"View Recipes ({len(st.session_state.bot.recipes)})"):
                    for key, recipe in st.session_state.bot.recipes.items():
                        st.markdown(f"**{recipe.get('name', key)}** ({recipe.get('region', 'Unknown')})")
    
    # Main chat interface
    if not st.session_state.api_key_set:
        st.info("👈 Please initialize the bot using the sidebar to start chatting!")
        return
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about cooking!"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.bot.chat(prompt)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()