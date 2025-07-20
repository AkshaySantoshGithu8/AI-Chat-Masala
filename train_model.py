# Suggested improvements for the Indian Recipe AI training

# 1. Expand Recipe Database
def get_expanded_recipe_database():
    """Add more diverse recipes across regions"""
    additional_recipes = [
        # Bengali
        {
            "name": "Fish Curry (Macher Jhol)",
            "region": "Bengali",
            "description": "Light and flavorful fish curry with minimal spices",
            "ingredients": ["Fish pieces", "Turmeric", "Mustard oil", "Ginger", "Green chilies"],
            "dietary_info": {"vegetarian": False, "spicy": "mild"}
        },
        # Gujarati
        {
            "name": "Dhokla",
            "region": "Gujarati",
            "description": "Steamed fermented cake made from chickpea flour",
            "ingredients": ["Chickpea flour", "Yogurt", "Ginger-green chili paste"],
            "dietary_info": {"vegetarian": True, "spicy": "mild"}
        },
        # Add more regional varieties...
    ]
    return additional_recipes

# 2. Add Dietary and Allergy Information
def add_dietary_context(recipe):
    """Enhance recipes with dietary information"""
    dietary_labels = []
    
    # Check for common dietary restrictions
    if "paneer" in str(recipe["ingredients"]).lower():
        dietary_labels.append("vegetarian")
    if "chicken" in str(recipe["ingredients"]).lower():
        dietary_labels.append("non-vegetarian")
    if "nuts" in str(recipe["ingredients"]).lower():
        dietary_labels.append("contains_nuts")
    
    recipe["dietary_info"] = dietary_labels
    return recipe

# 3. Add Nutritional Information
def add_nutrition_training():
    """Add training examples about nutrition"""
    nutrition_examples = [
        {
            "input": "Is dal makhani healthy?",
            "output": "Dal Makhani is rich in protein from lentils but high in calories due to cream and butter. It's a good source of fiber and plant-based protein. For a healthier version, you can reduce cream and add more vegetables.",
            "context": "nutrition"
        },
        {
            "input": "What are the health benefits of turmeric?",
            "output": "Turmeric contains curcumin, which has anti-inflammatory properties. It's commonly used in Indian cooking and may help with digestion and joint health. Fresh turmeric is more potent than powder.",
            "context": "health_benefits"
        }
    ]
    return nutrition_examples

# 4. Add Cooking Technique Training
def add_technique_training():
    """Add training for cooking techniques"""
    technique_examples = [
        {
            "input": "How do I make perfect tadka?",
            "output": "For perfect tadka (tempering): Heat oil/ghee until it shimmers. Add whole spices first (cumin, mustard seeds). When they splutter, add curry leaves and chopped aromatics. Pour immediately over dal while sizzling.",
            "context": "technique"
        },
        {
            "input": "What is bhuna technique?",
            "output": "Bhuna is a slow-cooking technique where ingredients are stirred and cooked over medium heat until the oil separates. This concentrates flavors and creates a rich, thick base for curries.",
            "context": "technique"
        }
    ]
    return technique_examples

# 5. Add Substitution Training
def add_substitution_training():
    """Add ingredient substitution knowledge"""
    substitution_examples = [
        {
            "input": "What can I substitute for heavy cream in butter chicken?",
            "output": "You can substitute heavy cream with: cashew cream (blend soaked cashews with water), coconut cream, greek yogurt (add at the end), or milk with a bit of cornstarch for thickness.",
            "context": "substitution"
        },
        {
            "input": "I don't have garam masala, what can I use?",
            "output": "Make quick garam masala by mixing: 1 tsp cumin powder, 1 tsp coriander powder, 1/2 tsp black pepper, 1/2 tsp cinnamon powder, 1/4 tsp cardamom powder, and 1/4 tsp clove powder.",
            "context": "substitution"
        }
    ]
    return substitution_examples

# 6. Enhanced Data Augmentation
def augment_training_data(base_examples):
    """Create variations of existing training examples"""
    augmented = []
    
    for example in base_examples:
        # Original example
        augmented.append(example)
        
        # Create variations with different phrasings
        variations = generate_question_variations(example["input"])
        for variation in variations:
            augmented.append({
                "input": variation,
                "output": example["output"],
                "context": example["context"]
            })
    
    return augmented

def generate_question_variations(original_question):
    """Generate different ways to ask the same question"""
    variations = []
    
    if "how do i make" in original_question.lower():
        recipe_name = original_question.lower().replace("how do i make", "").strip("?")
        variations.extend([
            f"Can you give me the recipe for {recipe_name}?",
            f"I want to cook {recipe_name}, how?",
            f"Show me how to prepare {recipe_name}",
            f"What's the recipe for {recipe_name}?"
        ])
    
    return variations

# 7. Add Error Handling and Validation
def validate_recipe_data(recipe):
    """Validate recipe data before training"""
    required_fields = ["name", "ingredients", "instructions", "region"]
    
    for field in required_fields:
        if field not in recipe:
            raise ValueError(f"Missing required field: {field}")
    
    # Validate ingredients list
    if not isinstance(recipe["ingredients"], list) or len(recipe["ingredients"]) < 3:
        raise ValueError("Recipe must have at least 3 ingredients")
    
    # Validate instructions
    if not isinstance(recipe["instructions"], list) or len(recipe["instructions"]) < 3:
        raise ValueError("Recipe must have at least 3 instruction steps")
    
    return True

# 8. Enhanced Model Configuration
def get_enhanced_training_config():
    """Optimized training configuration"""
    config = {
        "model_name": "microsoft/DialoGPT-medium",
        "max_length": 1024,  # Increased for longer recipes
        "learning_rate": 3e-5,  # Slightly lower for better convergence
        "num_epochs": 5,  # More epochs for better learning
        "batch_size": 4,  # Optimal for most GPUs
        "gradient_accumulation_steps": 2,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "save_strategy": "epoch",
        "eval_strategy": "epoch",
        "logging_steps": 50,
        "save_total_limit": 3,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "fp16": True,  # Enable mixed precision
        "dataloader_num_workers": 2,
        "remove_unused_columns": False,
        "report_to": "tensorboard",  # Enable tensorboard logging
    }
    return config

# 9. Add Model Evaluation Metrics
def evaluate_model_performance(model, tokenizer, test_dataset):
    """Evaluate model performance on test set"""
    from sklearn.metrics import accuracy_score
    import numpy as np
    
    predictions = []
    references = []
    
    for example in test_dataset:
        # Generate prediction
        input_text = f"User: {example['input']}\nAssistant:"
        inputs = tokenizer(input_text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=inputs.input_ids.shape[1] + 200,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(prediction)
        references.append(example['output'])
    
    # Calculate metrics (simplified)
    print(f"Generated {len(predictions)} predictions")
    print(f"Average prediction length: {np.mean([len(p.split()) for p in predictions])}")
    print(f"Average reference length: {np.mean([len(r.split()) for r in references])}")
    
    return {
        "num_predictions": len(predictions),
        "avg_prediction_length": np.mean([len(p.split()) for p in predictions]),
        "avg_reference_length": np.mean([len(r.split()) for r in references])
    }

# 10. Add Continuous Learning Support
def setup_continuous_learning():
    """Setup for continuous learning from user feedback"""
    feedback_examples = []
    
    def collect_feedback(query, response, rating, correction=None):
        """Collect user feedback for continuous improvement"""
        feedback = {
            "query": query,
            "response": response,
            "rating": rating,
            "timestamp": datetime.now().isoformat()
        }
        
        if correction:
            feedback["correction"] = correction
        
        feedback_examples.append(feedback)
        
        # Save feedback for later retraining
        with open("user_feedback.json", "a") as f:
            json.dump(feedback, f)
            f.write("\n")
    
    return collect_feedback